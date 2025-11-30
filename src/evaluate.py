#!/usr/bin/env python3
"""
Cross-Domain Evaluation: Test ALL models on REAL images
This addresses the teacher's critical feedback about synthetic-to-real transfer
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import json
import os

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
NUM_CLASSES = 3
CLASS_NAMES = ['no_leak', 'oil_leak', 'water_leak']

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_TEST_DIR = os.path.join(BASE_DIR, 'datasets/experiment_a/test')

# Models to evaluate
MODELS = {
    'Exp_A_Real': os.path.join(BASE_DIR, 'model_exp_a.pth'),
    'Exp_B_Synthetic': os.path.join(BASE_DIR, 'model_exp_b.pth'),
    'Exp_C_Hybrid': os.path.join(BASE_DIR, 'model_exp_c.pth'),
}

def load_model(model_path):
    """Load a trained ResNet50 model from checkpoint"""
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded from checkpoint (val_acc: {checkpoint.get('val_accuracy', 'N/A'):.2f}%)")
    else:
        model.load_state_dict(checkpoint)
    model = model.to(DEVICE)
    model.eval()
    return model

def get_real_test_loader():
    """Load the real test dataset"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(REAL_TEST_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"Real test set: {len(dataset)} images")
    print(f"Classes: {dataset.classes}")
    return loader, dataset.classes

def evaluate_model(model, loader):
    """Evaluate model and return predictions"""
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)

def main():
    print("=" * 60)
    print("CROSS-DOMAIN EVALUATION: All Models on REAL Test Images")
    print("=" * 60)
    loader, classes = get_real_test_loader()
    results = {}
    
    for name, model_path in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print("=" * 60)
        model = load_model(model_path)
        preds, labels = evaluate_model(model, loader)
        accuracy = (preds == labels).mean() * 100
        print(f"\n*** ACCURACY ON REAL IMAGES: {accuracy:.2f}% ***\n")
        report = classification_report(labels, preds, target_names=CLASS_NAMES, digits=4)
        print("Classification Report:")
        print(report)
        cm = confusion_matrix(labels, preds)
        print("Confusion Matrix:")
        print(f"{'':12} Pred_NL  Pred_Oil  Pred_Water")
        for i, row in enumerate(cm):
            print(f"{CLASS_NAMES[i]:12} {row[0]:7d}  {row[1]:8d}  {row[2]:10d}")
        results[name] = {
            'accuracy': accuracy,
            'report': classification_report(labels, preds, target_names=CLASS_NAMES, output_dict=True),
            'confusion_matrix': cm.tolist()
        }
    
    print("\n" + "=" * 60)
    print("SUMMARY: All Models on REAL Test Images")
    print("=" * 60)
    print(f"{'Model':<20} {'Accuracy':>10} {'Oil F1':>10} {'Water F1':>10} {'No-Leak F1':>10}")
    print("-" * 60)
    for name, res in results.items():
        acc = res['accuracy']
        oil_f1 = res['report']['oil_leak']['f1-score']
        water_f1 = res['report']['water_leak']['f1-score']
        noleak_f1 = res['report']['no_leak']['f1-score']
        print(f"{name:<20} {acc:>9.2f}% {oil_f1:>10.4f} {water_f1:>10.4f} {noleak_f1:>10.4f}")
    
    with open(os.path.join(BASE_DIR, 'cross_domain_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: cross_domain_results.json")

if __name__ == '__main__':
    main()
