#!/usr/bin/env python3
"""Create train/val/test splits for all three experiments."""

import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import random

random.seed(42)

REAL_DIR = Path("real_images")
SYNTHETIC_DIR = Path("synthetic_filtered")
OUTPUT_BASE = Path("datasets")

def create_experiment_a():
    """Experiment A: Real-only (1,051 images)"""
    print("\n" + "=" * 60)
    print("EXPERIMENT A: Real-Only Baseline")
    print("=" * 60)
    
    exp_dir = OUTPUT_BASE / "experiment_a"
    
    for class_name in ["no_leak", "oil_leak", "water_leak"]:
        images = list((REAL_DIR / class_name).glob("*.jpg"))
        
        # 70/15/15 split
        train, temp = train_test_split(images, test_size=0.3, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)
        
        for split, split_imgs in [("train", train), ("val", val), ("test", test)]:
            split_dir = exp_dir / split / class_name
            split_dir.mkdir(parents=True, exist_ok=True)
            for img in split_imgs:
                shutil.copy2(img, split_dir / img.name)
        
        print(f"  {class_name}: {len(images)} → {len(train)} train, {len(val)} val, {len(test)} test")
    
    total = sum(len(list((exp_dir / "train" / c).glob("*.jpg"))) for c in ["no_leak", "oil_leak", "water_leak"])
    print(f"  TOTAL train: {total}")

def create_experiment_b():
    """Experiment B: Synthetic-only (2,100 images, 700 per class)"""
    print("\n" + "=" * 60)
    print("EXPERIMENT B: Synthetic-Only")
    print("=" * 60)
    
    exp_dir = OUTPUT_BASE / "experiment_b"
    
    for class_name in ["no_leak", "oil_leak", "water_leak"]:
        images = sorted((SYNTHETIC_DIR / class_name).glob("*.jpg"))[:700]
        
        train, temp = train_test_split(images, test_size=0.3, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)
        
        for split, split_imgs in [("train", train), ("val", val), ("test", test)]:
            split_dir = exp_dir / split / class_name
            split_dir.mkdir(parents=True, exist_ok=True)
            for img in split_imgs:
                shutil.copy2(img, split_dir / img.name)
        
        print(f"  {class_name}: 700 → {len(train)} train, {len(val)} val, {len(test)} test")
    
    total = sum(len(list((exp_dir / "train" / c).glob("*.jpg"))) for c in ["no_leak", "oil_leak", "water_leak"])
    print(f"  TOTAL train: {total}")

def create_experiment_c():
    """Experiment C: Hybrid (1,051 real + 1,049 synthetic = 2,100)"""
    print("\n" + "=" * 60)
    print("EXPERIMENT C: Hybrid (Real + Synthetic)")
    print("=" * 60)
    
    exp_dir = OUTPUT_BASE / "experiment_c"
    
    # Target 700 per class
    targets = {
        "oil_leak": {"real": 212, "synthetic": 488},
        "water_leak": {"real": 440, "synthetic": 260},
        "no_leak": {"real": 399, "synthetic": 301}
    }
    
    for class_name, counts in targets.items():
        real_imgs = list((REAL_DIR / class_name).glob("*.jpg"))
        
        # Get synthetic (skip first 700 used in Exp B)
        all_synth = sorted((SYNTHETIC_DIR / class_name).glob("*.jpg"))
        synth_imgs = all_synth[700:700 + counts["synthetic"]]
        
        all_imgs = real_imgs + synth_imgs
        
        train, temp = train_test_split(all_imgs, test_size=0.3, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)
        
        for split, split_imgs in [("train", train), ("val", val), ("test", test)]:
            split_dir = exp_dir / split / class_name
            split_dir.mkdir(parents=True, exist_ok=True)
            for img in split_imgs:
                shutil.copy2(img, split_dir / img.name)
        
        print(f"  {class_name}: {len(real_imgs)} real + {len(synth_imgs)} synth = {len(all_imgs)}")
        print(f"           → {len(train)} train, {len(val)} val, {len(test)} test")
    
    total = sum(len(list((exp_dir / "train" / c).glob("*.jpg"))) for c in ["no_leak", "oil_leak", "water_leak"])
    print(f"  TOTAL train: {total}")

def main():
    print("=" * 60)
    print("CREATING EXPERIMENT DATASETS")
    print("=" * 60)
    
    # Clean previous
    if OUTPUT_BASE.exists():
        shutil.rmtree(OUTPUT_BASE)
    
    create_experiment_a()
    create_experiment_b()
    create_experiment_c()
    
    print("\n" + "=" * 60)
    print("✅ ALL DATASETS CREATED!")
    print("=" * 60)
    print("\nStructure:")
    print("  datasets/")
    print("    ├── experiment_a/  (real-only)")
    print("    ├── experiment_b/  (synthetic-only)")
    print("    └── experiment_c/  (hybrid)")
    print("=" * 60)

if __name__ == "__main__":
    main()
