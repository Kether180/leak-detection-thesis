#!/usr/bin/env python3
"""Filter synthetic images using trained ResNet18 classifier."""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import shutil
from tqdm import tqdm

# Configuration - LOWERED THRESHOLD
FILTER_MODEL = "filter_classifier_resnet18.pth"
SYNTHETIC_DIR = Path("synthetic_generation/experiment_b")
OUTPUT_DIR = Path("synthetic_filtered")
CONFIDENCE_THRESHOLD = 0.40  # Lowered from 0.70

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_filter_model(device):
    print(f"   Loading: {FILTER_MODEL}")
    checkpoint = torch.load(FILTER_MODEL, map_location=device)
    class_names = checkpoint['class_names']
    val_acc = checkpoint.get('val_accuracy', 0.0)
    
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"   Accuracy: {val_acc:.2f}%")
    print(f"   Classes: {class_names}")
    return model, class_names

def filter_class(class_name, model, class_names, device):
    input_dir = SYNTHETIC_DIR / class_name
    output_dir = OUTPUT_DIR / class_name
    
    # Clear previous filtered images
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    class_idx = class_names.index(class_name)
    images = sorted(input_dir.glob("*.jpg"))
    
    kept, rejected = 0, 0
    
    print(f"\n   Filtering {class_name}: {len(images)} images...")
    
    with torch.no_grad():
        for img_path in tqdm(images, desc=f"   {class_name}", ncols=70):
            try:
                image = Image.open(img_path).convert('RGB')
                tensor = transform(image).unsqueeze(0).to(device)
                
                probs = torch.softmax(model(tensor), dim=1)
                conf, pred = probs.max(1)
                
                if pred.item() == class_idx and conf.item() >= CONFIDENCE_THRESHOLD:
                    shutil.copy2(img_path, output_dir / img_path.name)
                    kept += 1
                else:
                    rejected += 1
            except Exception as e:
                print(f"\n   ERROR: {img_path.name}: {e}")
                rejected += 1
    
    rate = 100 * kept / (kept + rejected) if (kept + rejected) > 0 else 0
    print(f"   ✅ {class_name}: Kept {kept}/{kept+rejected} ({rate:.1f}%)")
    return kept, rejected

def main():
    print("=" * 60)
    print("FILTERING SYNTHETIC IMAGES (THRESHOLD: 0.40)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[1/3] Setup: {device}, threshold={CONFIDENCE_THRESHOLD}")
    
    print(f"\n[2/3] Loading model...")
    model, class_names = load_filter_model(device)
    
    print(f"\n[3/3] Filtering...")
    total_kept, total_rejected = 0, 0
    
    for class_name in ["no_leak", "oil_leak", "water_leak"]:
        kept, rejected = filter_class(class_name, model, class_names, device)
        total_kept += kept
        total_rejected += rejected
    
    rate = 100 * total_kept / (total_kept + total_rejected)
    
    print("\n" + "=" * 60)
    print(f"✅ COMPLETE: Kept {total_kept:,} / {total_kept+total_rejected:,} ({rate:.1f}%)")
    print(f"✅ Saved to: {OUTPUT_DIR}/")
    print("=" * 60)

if __name__ == "__main__":
    main()
