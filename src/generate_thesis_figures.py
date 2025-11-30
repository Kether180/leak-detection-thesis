#!/usr/bin/env python3
"""
Generate Figure Grids for Thesis
Creates professional image grids showing real and synthetic examples
"""

import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_DIR = os.path.join(BASE_DIR, 'real_images')
SYNTHETIC_DIR = os.path.join(BASE_DIR, 'synthetic_filtered')
OUTPUT_DIR = os.path.join(BASE_DIR, 'thesis_figures')

CLASS_NAMES = ['no_leak', 'oil_leak', 'water_leak']
CLASS_LABELS = ['No-Leak (Clean Equipment)', 'Oil Leak', 'Water Leak']

# Set seed for reproducibility
random.seed(42)

def get_random_images(directory, class_name, n=3):
    """Get n random images from a class directory"""
    class_dir = os.path.join(directory, class_name)
    if not os.path.exists(class_dir):
        print(f"Warning: {class_dir} not found")
        return []
    images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    selected = random.sample(images, min(n, len(images)))
    return [os.path.join(class_dir, img) for img in selected]

def create_dataset_grid(image_dir, title, output_name, samples_per_class=3):
    """Create a 3x3 grid of dataset examples"""
    fig, axes = plt.subplots(3, samples_per_class, figsize=(12, 12))
    
    for row, (class_name, class_label) in enumerate(zip(CLASS_NAMES, CLASS_LABELS)):
        images = get_random_images(image_dir, class_name, samples_per_class)
        
        for col in range(samples_per_class):
            ax = axes[row, col]
            
            if col < len(images):
                img = Image.open(images[col]).convert('RGB')
                img = img.resize((224, 224))
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=14)
            
            ax.axis('off')
            
            # Row labels on the left
            if col == 0:
                ax.set_ylabel(class_label, fontsize=12, fontweight='bold', labelpad=10)
                ax.yaxis.set_label_position("left")
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, output_name)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {output_path}")

def create_comparison_grid():
    """Create side-by-side real vs synthetic comparison"""
    fig, axes = plt.subplots(3, 6, figsize=(18, 10))
    
    for row, (class_name, class_label) in enumerate(zip(CLASS_NAMES, CLASS_LABELS)):
        # Real images (columns 0-2)
        real_images = get_random_images(REAL_DIR, class_name, 3)
        for col in range(3):
            ax = axes[row, col]
            if col < len(real_images):
                img = Image.open(real_images[col]).convert('RGB').resize((224, 224))
                ax.imshow(img)
            ax.axis('off')
            if row == 0 and col == 1:
                ax.set_title('REAL IMAGES', fontsize=14, fontweight='bold', pad=10)
        
        # Synthetic images (columns 3-5)
        synth_images = get_random_images(SYNTHETIC_DIR, class_name, 3)
        for col in range(3):
            ax = axes[row, col + 3]
            if col < len(synth_images):
                img = Image.open(synth_images[col]).convert('RGB').resize((224, 224))
                ax.imshow(img)
            ax.axis('off')
            if row == 0 and col == 1:
                ax.set_title('SYNTHETIC IMAGES', fontsize=14, fontweight='bold', pad=10)
        
        # Row label
        axes[row, 0].set_ylabel(class_label, fontsize=11, fontweight='bold', labelpad=10)
    
    # Add vertical separator line
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    
    plt.suptitle('Real vs Synthetic Image Comparison', fontsize=16, fontweight='bold', y=1.02)
    
    output_path = os.path.join(OUTPUT_DIR, 'real_vs_synthetic_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {output_path}")

def create_results_bar_chart():
    """Create bar chart comparing model accuracies"""
    models = ['Real-Only\n(Exp A)', 'Synthetic-Only\n(Exp B)', 'Hybrid\n(Exp C)']
    accuracies = [97.47, 47.47, 99.37]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Accuracy on Real Test Images (%)', fontsize=12)
    ax.set_title('Cross-Domain Evaluation: All Models Tested on Real Images', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5, label='Target: 90%')
    ax.legend()
    
    # Add grid
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'accuracy_comparison_chart.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {output_path}")

def create_f1_comparison_chart():
    """Create grouped bar chart comparing F1 scores per class"""
    classes = ['Oil Leak', 'Water Leak', 'No-Leak']
    
    real_f1 = [0.9677, 0.9706, 0.9831]
    synth_f1 = [0.2667, 0.5985, 0.4615]
    hybrid_f1 = [0.9841, 1.0000, 0.9917]
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width, real_f1, width, label='Real-Only (Exp A)', color='#3498db')
    bars2 = ax.bar(x, synth_f1, width, label='Synthetic-Only (Exp B)', color='#e74c3c')
    bars3 = ax.bar(x + width, hybrid_f1, width, label='Hybrid (Exp C)', color='#2ecc71')
    
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('Per-Class F1-Score Comparison on Real Test Images', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.15)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'f1_comparison_chart.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {output_path}")

def main():
    print("=" * 60)
    print("GENERATING THESIS FIGURES")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n1. Creating real image examples grid...")
    create_dataset_grid(REAL_DIR, 'Real Image Dataset Examples', 'fig_real_examples.png')
    
    print("\n2. Creating synthetic image examples grid...")
    create_dataset_grid(SYNTHETIC_DIR, 'Synthetic Image Dataset Examples', 'fig_synthetic_examples.png')
    
    print("\n3. Creating real vs synthetic comparison...")
    create_comparison_grid()
    
    print("\n4. Creating accuracy comparison chart...")
    create_results_bar_chart()
    
    print("\n5. Creating F1-score comparison chart...")
    create_f1_comparison_chart()
    
    print(f"\n{'='*60}")
    print(f"✅ All figures saved to: {OUTPUT_DIR}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
