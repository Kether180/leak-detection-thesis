#!/usr/bin/env python3
"""
Generate Additional Thesis Figures:
- Confusion matrix heatmaps
- Training curves (loss & accuracy)
- Filtering acceptance rate chart
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'thesis_figures')

CLASS_NAMES = ['No-Leak', 'Oil Leak', 'Water Leak']

# Confusion matrices from cross-domain evaluation
CONFUSION_MATRICES = {
    'Exp A: Real-Only': np.array([
        [58, 0, 2],
        [0, 30, 2],
        [0, 0, 66]
    ]),
    'Exp B: Synthetic-Only': np.array([
        [24, 8, 28],
        [20, 10, 2],
        [0, 25, 41]
    ]),
    'Exp C: Hybrid': np.array([
        [60, 0, 0],
        [1, 31, 0],
        [0, 0, 66]
    ])
}

ACCURACIES = {
    'Exp A: Real-Only': 97.47,
    'Exp B: Synthetic-Only': 47.47,
    'Exp C: Hybrid': 99.37
}

def create_confusion_matrix_figure():
    """Create side-by-side confusion matrices for all experiments"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    for idx, (name, cm) in enumerate(CONFUSION_MATRICES.items()):
        ax = axes[idx]
        
        # Normalize for color intensity
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    cbar=False, annot_kws={'size': 14, 'fontweight': 'bold'})
        
        ax.set_xlabel('Predicted Label', fontsize=11)
        ax.set_ylabel('True Label', fontsize=11)
        acc = ACCURACIES[name]
        ax.set_title(f'{name}\nAccuracy: {acc:.2f}%', fontsize=12, fontweight='bold')
        
        # Rotate labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    plt.suptitle('Confusion Matrices: All Models Evaluated on Real Test Images', 
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'confusion_matrices_all.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {output_path}")

def create_individual_confusion_matrices():
    """Create individual confusion matrix figures"""
    for name, cm in CONFUSION_MATRICES.items():
        fig, ax = plt.subplots(figsize=(6, 5))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    annot_kws={'size': 16, 'fontweight': 'bold'})
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        acc = ACCURACIES[name]
        ax.set_title(f'{name}\nAccuracy: {acc:.2f}%', fontsize=14, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Clean filename
        filename = name.lower().replace(' ', '_').replace(':', '').replace('-', '_')
        output_path = os.path.join(OUTPUT_DIR, f'confusion_matrix_{filename}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ Saved: {output_path}")

def create_training_curves():
    """Create training curves from history JSON files"""
    histories = {}
    exp_names = ['a', 'b', 'c']
    exp_labels = ['Exp A: Real-Only', 'Exp B: Synthetic-Only', 'Exp C: Hybrid']
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    # Load histories
    for exp in exp_names:
        history_path = os.path.join(BASE_DIR, f'model_exp_{exp}_history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                histories[exp] = json.load(f)
    
    if not histories:
        print("⚠️ No training history files found")
        return
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot training loss
    ax1 = axes[0]
    for exp, label, color in zip(exp_names, exp_labels, colors):
        if exp in histories:
            epochs = range(1, len(histories[exp]['train_loss']) + 1)
            ax1.plot(epochs, histories[exp]['train_loss'], label=label, color=color, linewidth=2)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss Over Epochs', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, 35)
    
    # Plot validation accuracy
    ax2 = axes[1]
    for exp, label, color in zip(exp_names, exp_labels, colors):
        if exp in histories:
            epochs = range(1, len(histories[exp]['val_acc']) + 1)
            ax2.plot(epochs, histories[exp]['val_acc'], label=label, color=color, linewidth=2)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax2.set_title('Validation Accuracy Over Epochs', fontsize=13, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, 35)
    ax2.set_ylim(50, 100)
    
    plt.suptitle('Training Progress Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'training_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {output_path}")

def create_filtering_acceptance_chart():
    """Create chart showing differential filtering acceptance rates"""
    classes = ['No-Leak', 'Oil Leak', 'Water Leak']
    acceptance_rates = [82.7, 8.8, 37.4]
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(classes, acceptance_rates, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, rate in zip(bars, acceptance_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Filter Acceptance Rate (%)', fontsize=12)
    ax.set_xlabel('Image Class', fontsize=12)
    ax.set_title('Automated Filtering: Differential Acceptance Rates\n(ResNet18 Filter Trained on Real Images)', 
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, 100)
    
    # Add horizontal line at 50%
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
    
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    # Add annotation
    ax.text(0.5, -0.15, 
            'Note: Low acceptance for oil/water leaks indicates Stable Diffusion struggles\nto generate realistic fluid dynamics compared to static equipment.',
            transform=ax.transAxes, ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'filtering_acceptance_rates.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {output_path}")

def create_dataset_composition_chart():
    """Create stacked bar chart showing dataset composition"""
    experiments = ['Exp A\n(Real-Only)', 'Exp B\n(Synthetic-Only)', 'Exp C\n(Hybrid)']
    
    real_counts = [1051, 0, 1051]
    synth_counts = [0, 2100, 1049]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(experiments))
    width = 0.5
    
    bars1 = ax.bar(x, real_counts, width, label='Real Images', color='#3498db')
    bars2 = ax.bar(x, synth_counts, width, bottom=real_counts, label='Synthetic Images', color='#e74c3c')
    
    ax.set_ylabel('Number of Images', fontsize=12)
    ax.set_xlabel('Experiment', fontsize=12)
    ax.set_title('Dataset Composition by Experiment', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, fontsize=11)
    ax.legend(loc='upper right')
    
    # Add totals on top
    totals = [r + s for r, s in zip(real_counts, synth_counts)]
    for i, total in enumerate(totals):
        ax.text(i, total + 50, f'{total}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylim(0, 2500)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'dataset_composition.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {output_path}")

def create_domain_gap_visualization():
    """Visualize the synthetic-to-real domain gap"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data
    categories = ['Exp B on\nSynthetic Test', 'Exp B on\nReal Test']
    accuracies = [86.03, 47.47]
    colors = ['#f39c12', '#e74c3c']
    
    bars = ax.bar(categories, accuracies, color=colors, edgecolor='black', linewidth=2, width=0.5)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                f'{acc:.1f}%', ha='center', fontsize=16, fontweight='bold')
    
    # Draw gap arrow
    ax.annotate('', xy=(1, 47.47), xytext=(0, 86.03),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax.text(0.5, 67, 'Domain Gap:\n38.6%', ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Synthetic-to-Real Domain Gap\n(Experiment B: Synthetic-Only Model)', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'domain_gap_visualization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {output_path}")

def main():
    print("=" * 60)
    print("GENERATING ADDITIONAL THESIS FIGURES")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n1. Creating confusion matrices (combined)...")
    create_confusion_matrix_figure()
    
    print("\n2. Creating individual confusion matrices...")
    create_individual_confusion_matrices()
    
    print("\n3. Creating training curves...")
    create_training_curves()
    
    print("\n4. Creating filtering acceptance chart...")
    create_filtering_acceptance_chart()
    
    print("\n5. Creating dataset composition chart...")
    create_dataset_composition_chart()
    
    print("\n6. Creating domain gap visualization...")
    create_domain_gap_visualization()
    
    print(f"\n{'='*60}")
    print(f"✅ All additional figures saved to: {OUTPUT_DIR}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
