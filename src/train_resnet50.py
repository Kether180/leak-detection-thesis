import os
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from PIL import Image


# ---- SAFE IMAGE LOADER ----
class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        # Try up to 100 different indices if one is corrupted
        # This gives a much better chance of finding valid images
        for _ in range(100):
            try:
                return super().__getitem__(index)
            except Exception:
                index = (index + 1) % len(self.samples)
        # Still failing after 100 tries → return dummy item
        return None, None


# ---- CUSTOM COLLATE FUNCTION ----
def collate_fn_filter_none(batch):
    """Remove None samples from batch before collating"""
    batch = [item for item in batch if item[0] is not None]
    if len(batch) == 0:
        return None, None
    return torch.utils.data.dataloader.default_collate(batch)


# ---- CONFIG ----
OUTPUT_DIR = "outputs"
DATA_DIR = "dataset"
BATCH_SIZE = 32
EPOCHS = 35
LEARNING_RATE = 3e-4
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_PATH = os.path.join(OUTPUT_DIR, "leak_detector_best.pt")

# Reproducibility
random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training using device: {device}")


# ---- TRANSFORMS ----
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(12),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2),
    transforms.RandomPerspective(distortion_scale=0.25, p=0.4),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

test_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ---- DATASETS ----
train_ds = SafeImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
val_ds   = SafeImageFolder(os.path.join(DATA_DIR, "val"),   transform=test_tf)
test_ds  = SafeImageFolder(os.path.join(DATA_DIR, "test"),  transform=test_tf)

print(f"Dataset loaded: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test")


# ---- CLASS BALANCING ----
class_counts = torch.tensor([
    sum(1 for _, cls in train_ds.samples if cls == i)
    for i in range(len(train_ds.classes))
])
class_weights = 1.0 / class_counts
sample_weights = [class_weights[cls] for _, cls in train_ds.samples]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, 
                      num_workers=0, collate_fn=collate_fn_filter_none)
val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, 
                      num_workers=0, collate_fn=collate_fn_filter_none)
test_dl  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, 
                      num_workers=0, collate_fn=collate_fn_filter_none)

classes = train_ds.classes
print("Classes:", classes)


# ---- MODEL ----
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scaler = torch.cuda.amp.GradScaler()
best_acc = 0.0


# ---- TRAINING ----
def train_model():
    global best_acc

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 30)

        for phase in ("train", "val"):
            model.train() if phase == "train" else model.eval()
            loader = train_dl if phase == "train" else val_dl

            running_loss = 0.0
            running_corrects = 0
            samples_processed = 0
            corrupted_batches = 0

            print(f"→ Starting {phase} dataloader with {len(loader.dataset)} images")

            for batch in loader:
                inputs, labels = batch

                # Skip empty batches (all samples were corrupted)
                if inputs is None or labels is None:
                    corrupted_batches += 1
                    continue

                inputs, labels = inputs.to(device), labels.to(device)
                batch_size = inputs.size(0)
                samples_processed += batch_size
                
                optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(phase == "train"), torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                if phase == "train":
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                running_loss += loss.item() * batch_size
                running_corrects += (preds == labels).sum()

            # Calculate metrics based on actual samples processed
            if samples_processed > 0:
                epoch_loss = running_loss / samples_processed
                epoch_acc = running_corrects.double() / samples_processed
                
                skipped_samples = len(loader.dataset) - samples_processed
                print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
                print(f"  → Processed: {samples_processed}/{len(loader.dataset)} samples")
                if corrupted_batches > 0:
                    print(f"  → Skipped: {corrupted_batches} corrupted batches (~{corrupted_batches * BATCH_SIZE} images)")

                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), MODEL_PATH)
                    print("✔ Best model updated!")
            else:
                print(f"[ERROR] No valid samples processed in {phase} phase!")
                print(f"  → All {corrupted_batches} batches were corrupted!")

    print(f"\nTraining complete. Best validation accuracy: {best_acc:.4f}")


# ---- EVALUATION ----
def evaluate():
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in test_dl:
            if inputs is None:
                continue

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=classes))

    report_str = classification_report(y_true, y_pred, target_names=classes)
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write(report_str)

    cm = confusion_matrix(y_true, y_pred)
    np.savetxt(os.path.join(OUTPUT_DIR, "confusion_matrix.txt"), cm, fmt="%d")
    print("Saved confusion matrix to outputs/confusion_matrix.txt")


# ---- MAIN ----
if __name__ == "__main__":
    train_model()
    evaluate()
