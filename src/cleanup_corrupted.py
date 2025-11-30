#!/usr/bin/env python3
"""Find and remove corrupted images that can't be opened."""

from PIL import Image
from pathlib import Path
import os

REAL_IMAGES_DIR = Path("real_images")

def check_and_clean():
    print("=" * 60)
    print("Scanning for corrupted images...")
    print("=" * 60)
    
    corrupted = []
    valid = 0
    
    for class_dir in REAL_IMAGES_DIR.iterdir():
        if not class_dir.is_dir():
            continue
            
        print(f"\nChecking {class_dir.name}...")
        
        for img_path in class_dir.glob("*.jpg"):
            try:
                with Image.open(img_path) as img:
                    img.verify()  # Verify it's a valid image
                valid += 1
            except Exception as e:
                print(f"  ❌ CORRUPTED: {img_path.name}")
                corrupted.append(img_path)
    
    print("\n" + "=" * 60)
    print(f"Valid images: {valid}")
    print(f"Corrupted images: {len(corrupted)}")
    print("=" * 60)
    
    if corrupted:
        print("\nRemoving corrupted images...")
        for path in corrupted:
            print(f"  Removing: {path}")
            os.remove(path)
        print(f"\n✅ Removed {len(corrupted)} corrupted images")
    else:
        print("\n✅ No corrupted images found!")
    
    # Final count
    print("\n" + "=" * 60)
    print("FINAL IMAGE COUNTS:")
    print("=" * 60)
    total = 0
    for class_dir in sorted(REAL_IMAGES_DIR.iterdir()):
        if class_dir.is_dir():
            count = len(list(class_dir.glob("*.jpg")))
            total += count
            print(f"  {class_dir.name}: {count} images")
    print(f"  TOTAL: {total} images")
    print("=" * 60)

if __name__ == "__main__":
    check_and_clean()
