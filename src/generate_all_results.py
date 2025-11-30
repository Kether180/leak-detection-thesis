import os
import re
import random
import torch
import numpy as np

import matplotlib
matplotlib.use("Agg")  # <<< CRITICAL — fixes freeze
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from torchvision import models, transforms, datasets
from torchvision.utils import make_grid
from sklearn.metrics import confusion_matrix, classification_report

OUTPUT_DIR = "outputs"
DATA_DIR = "dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- 1) Read latest training log ----------
log_files = sorted([f for f in os.listdir(".") if f.startswith("train_") and f.endswith(".log")])
LOG_PATH = log_files[-1]
print(f"📌 Using log file: {LOG_PATH}")

train_loss, val_loss, train_acc, val_acc = [], [], [], []
with open(LOG_PATH, "r") as f:
    for line in f:
        t = re.search(r"train Loss: ([0-9.]+) Acc: ([0-9.]+)", line)
        v = re.search(r"val Loss: ([0-9.]+) Acc: ([0-9.]+)", line)
        if t:
            train_loss.append(float(t.group(1)))
            train_acc.append(float(t.group(2)))
        if v:
            val_loss.append(float(v.group(1)))
            val_acc.append(float(v.group(2)))

epochs = list(range(1, len(train_loss) + 1))

plt.plot(epochs, train_acc, label="Train")
plt.plot(epochs, val_acc, label="Validation")
plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.title("Accuracy Curve"); plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "training_accuracy.png"))
plt.close()

plt.plot(epochs, train_loss, label="Train")
plt.plot(epochs, val_loss, label="Validation")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("Loss Curve"); plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "training_loss.png"))
plt.close()

print("✔ Saved training curves")


# ---------- 2) Load model and test dataset ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(OUTPUT_DIR, "leak_detector_best.pt")

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

test_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=val_tf)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)
classes = test_ds.classes

model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device); model.eval()


# ---------- 3) Confusion Matrix ----------
print("Computing confusion matrix...")
y_true, y_pred = [], []
with torch.no_grad():
    for x, y in test_dl:
        x, y = x.to(device), y.to(device)
        preds = torch.argmax(model(x), 1)
        y_true.extend(y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()
print("✔ Saved confusion matrix")


# ---------- 4) Sample Prediction Grid ----------
print("Generating prediction preview grid...")
samples = random.sample(test_ds.samples, k=min(8, len(test_ds)))
images, _ = zip(*samples)

tensors = []
for img in images:
    try:
        tensors.append(val_tf(Image.open(img)))
    except:
        pass

tensor_batch = torch.stack(tensors).to(device)
preds = torch.argmax(model(tensor_batch), 1).cpu().numpy()

grid = make_grid(tensor_batch.cpu(), nrow=4, normalize=True)
plt.figure(figsize=(9, 6))
plt.imshow(grid.permute(1,2,0)); plt.axis("off")
plt.title("Predictions: " + ", ".join([classes[p] for p in preds]))
plt.savefig(os.path.join(OUTPUT_DIR, "sample_predictions_grid.png"))
plt.close()

print("\n🎉 ALL DONE! Saved to outputs/:")
print("   • training_accuracy.png")
print("   • training_loss.png")
print("   • confusion_matrix.png")
print("   • sample_predictions_grid.png")

