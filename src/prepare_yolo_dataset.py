"""
Prepare dataset for YOLOv8 object detection training.

This script:
1. Copies real images to YOLO directory structure
2. Creates empty label files (to be annotated)
3. Splits into train/val sets (80/20)

After running this script, annotate images using:
- Label Studio (local): https://labelstud.io/
- Roboflow (cloud): https://roboflow.com/
- CVAT (local/cloud): https://cvat.ai/

YOLO label format (one .txt file per image):
class_id x_center y_center width height
- All values normalized to [0, 1]
- class_id: 0=oil_leak, 1=water_leak (no_leak images have empty labels)
"""

import os
import shutil
import random
from pathlib import Path

# Configuration
RANDOM_SEED = 42
VAL_SPLIT = 0.2  # 20% for validation

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
SOURCE_DIR = DATA_ROOT / "leaks_organized"
YOLO_DIR = DATA_ROOT / "yolo"

# Class mapping for YOLO (only leak classes need bboxes)
# no_leak images will have empty label files
CLASS_MAPPING = {
    "oil_leak": 0,
    "water_leak": 1,
}

def collect_real_images():
    """Collect all real images from each class."""
    images = {
        "oil_leak": [],
        "water_leak": [],
        "no_leak": []
    }

    for class_name in images.keys():
        class_dir = SOURCE_DIR / class_name / "real"
        if class_dir.exists():
            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    images[class_name].append(img_path)

    return images

def split_dataset(images, val_ratio=0.2):
    """Split images into train and validation sets."""
    random.seed(RANDOM_SEED)

    train_images = []
    val_images = []

    for class_name, img_list in images.items():
        random.shuffle(img_list)
        split_idx = int(len(img_list) * (1 - val_ratio))

        train_images.extend([(img, class_name) for img in img_list[:split_idx]])
        val_images.extend([(img, class_name) for img in img_list[split_idx:]])

    return train_images, val_images

def copy_images_and_create_labels(image_list, split_name):
    """Copy images and create empty label files."""
    images_dir = YOLO_DIR / "images" / split_name
    labels_dir = YOLO_DIR / "labels" / split_name

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    stats = {"oil_leak": 0, "water_leak": 0, "no_leak": 0}

    for img_path, class_name in image_list:
        # Create unique filename with class prefix
        new_filename = f"{class_name}_{img_path.name}"

        # Copy image
        dst_img = images_dir / new_filename
        shutil.copy2(img_path, dst_img)

        # Create empty label file (to be annotated)
        label_filename = new_filename.rsplit(".", 1)[0] + ".txt"
        label_path = labels_dir / label_filename

        # Create empty label file with comment header
        with open(label_path, "w") as f:
            if class_name == "no_leak":
                # No leak images should remain empty (no objects to detect)
                f.write("")
            else:
                # Leak images need annotation - add placeholder comment
                f.write(f"# TODO: Annotate {class_name} bounding box(es)\n")
                f.write(f"# Format: class_id x_center y_center width height\n")
                f.write(f"# class_id: 0=oil_leak, 1=water_leak\n")
                f.write(f"# All values normalized to [0, 1]\n")

        stats[class_name] += 1

    return stats

def main():
    print("=" * 60)
    print("Preparing YOLOv8 Dataset for Leak Detection")
    print("=" * 60)

    # Collect images
    print("\n[1/4] Collecting real images...")
    images = collect_real_images()

    total = sum(len(v) for v in images.values())
    print(f"   Found {total} real images:")
    for class_name, img_list in images.items():
        print(f"   - {class_name}: {len(img_list)}")

    # Split dataset
    print(f"\n[2/4] Splitting dataset ({int((1-VAL_SPLIT)*100)}/{int(VAL_SPLIT*100)} train/val)...")
    train_images, val_images = split_dataset(images, VAL_SPLIT)
    print(f"   Train: {len(train_images)} images")
    print(f"   Val: {len(val_images)} images")

    # Copy images and create labels
    print("\n[3/4] Copying training images...")
    train_stats = copy_images_and_create_labels(train_images, "train")
    print(f"   Copied: oil_leak={train_stats['oil_leak']}, water_leak={train_stats['water_leak']}, no_leak={train_stats['no_leak']}")

    print("\n[4/4] Copying validation images...")
    val_stats = copy_images_and_create_labels(val_images, "val")
    print(f"   Copied: oil_leak={val_stats['oil_leak']}, water_leak={val_stats['water_leak']}, no_leak={val_stats['no_leak']}")

    # Summary
    print("\n" + "=" * 60)
    print("Dataset Preparation Complete!")
    print("=" * 60)
    print(f"\nOutput directory: {YOLO_DIR}")
    print(f"\nStructure:")
    print(f"  data/yolo/")
    print(f"  ├── images/")
    print(f"  │   ├── train/  ({len(train_images)} images)")
    print(f"  │   └── val/    ({len(val_images)} images)")
    print(f"  └── labels/")
    print(f"      ├── train/  (placeholder .txt files)")
    print(f"      └── val/    (placeholder .txt files)")

    print("\n" + "=" * 60)
    print("NEXT STEPS: Annotate Images")
    print("=" * 60)
    print("""
1. OPTION A - Roboflow (Recommended for beginners):
   - Go to https://roboflow.com/
   - Create project → Object Detection
   - Upload images from data/yolo/images/train/
   - Draw bounding boxes around leaks
   - Export as "YOLOv8" format
   - Replace data/yolo/labels/ with exported labels

2. OPTION B - Label Studio (Local, free):
   pip install label-studio
   label-studio start
   - Create project → Object Detection with Bounding Boxes
   - Import images from data/yolo/images/train/
   - Annotate and export as YOLO format

3. OPTION C - CVAT (Powerful, free):
   - Go to https://app.cvat.ai/
   - Create task → Images
   - Upload and annotate
   - Export as YOLO 1.1 format

Classes to annotate:
  - 0: oil_leak (dark/black fluid patches)
  - 1: water_leak (clear/reflective fluid patches)
  - no_leak images: Leave empty (no bounding boxes)

After annotation, run:
  python src/train_yolo.py
""")

if __name__ == "__main__":
    main()
