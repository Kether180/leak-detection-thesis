#!/usr/bin/env python3
"""Grad-CAM Visualization"""

import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 3
CLASS_NAMES = ['no_leak', 'oil_leak', 'water_leak']
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_TEST_DIR = os.path.join(BASE_DIR, 'datasets/experiment_a/test')
OUTPUT_DIR = os.path.join(BASE_DIR, 'gradcam_outputs')
MODELS = {
    'Exp_A_Real': os.path.join(BASE_DIR, 'model_exp_a.pth'),
    'Exp_B_Synthetic': os.path.join(BASE_DIR, 'model_exp_b.pth'),
    'Exp_C_Hybrid': os.path.join(BASE_DIR, 'model_exp_c.pth'),
}

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(lambda m, i, o: setattr(self, 'activations', o.detach()))
        target_layer.register_full_backward_hook(lambda m, gi, go: setattr(self, 'gradients', go[0].detach()))
    
    def generate(self, input_tensor, target_class=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        output[0, target_class].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.squeeze().cpu().numpy(), target_class, output.softmax(dim=1)[0, target_class].item()

def load_model(model_path):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(DEVICE).eval()
    return model

def overlay_heatmap(img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return np.uint8(alpha * heatmap + (1 - alpha) * img)

def main():
    print("=" * 60)
    print("GRAD-CAM VISUALIZATION")
    print("=" * 60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    row_labels = ['No-Leak', 'Oil Leak', 'Water Leak']
    col_labels = ['Original', 'Real-Only', 'Synth-Only', 'Hybrid']
    
    for row, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(REAL_TEST_DIR, class_name)
        img_path = os.path.join(class_dir, sorted(os.listdir(class_dir))[0])
        orig_img = Image.open(img_path).convert('RGB').resize((224, 224))
        img_array = np.array(orig_img)
        input_tensor = transform(orig_img).unsqueeze(0).to(DEVICE)
        
        axes[row, 0].imshow(img_array)
        axes[row, 0].set_ylabel(row_labels[row], fontsize=12, fontweight='bold')
        if row == 0: axes[row, 0].set_title(col_labels[0], fontsize=12, fontweight='bold')
        axes[row, 0].axis('off')
        
        for col, (name, model_path) in enumerate(MODELS.items()):
            model = load_model(model_path)
            gradcam = GradCAM(model, model.layer4[-1])
            heatmap, pred_class, conf = gradcam.generate(input_tensor)
            overlay = overlay_heatmap(img_array, heatmap)
            axes[row, col + 1].imshow(overlay)
            if row == 0: axes[row, col + 1].set_title(col_labels[col + 1], fontsize=12, fontweight='bold')
            pred_name = CLASS_NAMES[pred_class]
            color = 'green' if pred_name == class_name else 'red'
            axes[row, col + 1].text(5, 20, f'{pred_name[:3]} {conf:.0%}', color='white', fontsize=9,
                                    fontweight='bold', bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
            axes[row, col + 1].axis('off')
    
    plt.suptitle('Grad-CAM: Model Attention on Real Test Images', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'gradcam_summary_grid.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {OUTPUT_DIR}/gradcam_summary_grid.png")

if __name__ == '__main__':
    main()
