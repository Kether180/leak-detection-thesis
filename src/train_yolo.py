"""
Train YOLOv8 for leak detection (oil_leak, water_leak).

This script trains a YOLOv8 model on annotated leak detection data.
Before running, ensure you have:
1. Run prepare_yolo_dataset.py to set up the directory structure
2. Annotated the images with bounding boxes (Roboflow, Label Studio, or CVAT)
3. Placed the YOLO-format labels in data/yolo/labels/

Usage:
    python src/train_yolo.py                    # Default: yolov8m, 100 epochs
    python src/train_yolo.py --model yolov8s   # Smaller model (faster)
    python src/train_yolo.py --model yolov8l   # Larger model (more accurate)
    python src/train_yolo.py --epochs 50       # Fewer epochs
    python src/train_yolo.py --resume          # Resume from last checkpoint
"""

import argparse
import os
from pathlib import Path
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 for leak detection")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8m.pt",
        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        help="YOLOv8 model variant (n=nano, s=small, m=medium, l=large, x=xlarge)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to use (cuda:0, cpu, or empty for auto)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of data loader workers"
    )
    return parser.parse_args()

def check_dataset():
    """Verify dataset exists and has annotations."""
    project_root = Path(__file__).parent.parent
    yolo_dir = project_root / "data" / "yolo"

    train_images = yolo_dir / "images" / "train"
    train_labels = yolo_dir / "labels" / "train"

    if not train_images.exists():
        print("ERROR: Training images not found!")
        print(f"Expected: {train_images}")
        print("\nRun first: python src/prepare_yolo_dataset.py")
        return False

    # Count images and labels
    image_files = list(train_images.glob("*.[jp][pn][g]"))
    label_files = list(train_labels.glob("*.txt"))

    if len(image_files) == 0:
        print("ERROR: No training images found!")
        print("\nRun first: python src/prepare_yolo_dataset.py")
        return False

    # Check if labels have actual annotations (not just comments)
    annotated_count = 0
    for label_file in label_files:
        with open(label_file, "r") as f:
            content = f.read().strip()
            # Skip comment lines and empty files (no_leak)
            lines = [l for l in content.split("\n") if l and not l.startswith("#")]
            if lines:
                annotated_count += 1

    # We expect at least some images to be annotated (oil_leak and water_leak)
    expected_leak_images = len([f for f in image_files if "oil_leak" in f.name or "water_leak" in f.name])

    print(f"Dataset check:")
    print(f"  - Training images: {len(image_files)}")
    print(f"  - Label files: {len(label_files)}")
    print(f"  - Annotated (non-empty): {annotated_count}")
    print(f"  - Expected leak images: {expected_leak_images}")

    if annotated_count == 0:
        print("\nWARNING: No annotations found!")
        print("Labels contain only comments or are empty.")
        print("\nPlease annotate your images first:")
        print("  1. Use Roboflow, Label Studio, or CVAT")
        print("  2. Export as YOLO format")
        print("  3. Place labels in data/yolo/labels/train/")
        return False

    if annotated_count < expected_leak_images * 0.5:
        print(f"\nWARNING: Only {annotated_count}/{expected_leak_images} leak images annotated.")
        print("Consider annotating more images for better results.")
        response = input("Continue anyway? [y/N]: ")
        if response.lower() != "y":
            return False

    return True

def main():
    args = parse_args()

    # Import ultralytics (YOLOv8)
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed!")
        print("Run: pip install ultralytics")
        return

    # Check dataset
    print("=" * 60)
    print("YOLOv8 Training for Leak Detection")
    print("=" * 60)

    if not check_dataset():
        return

    # Paths
    project_root = Path(__file__).parent.parent
    config_path = project_root / "configs" / "leak_dataset.yaml"

    # Create run name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"leak_detector_{timestamp}"

    # Load model
    print(f"\n[1/3] Loading model: {args.model}")
    model = YOLO(args.model)

    # Training configuration
    print(f"\n[2/3] Starting training...")
    print(f"  - Model: {args.model}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch}")
    print(f"  - Image size: {args.imgsz}")
    print(f"  - Device: {args.device if args.device else 'auto'}")

    # Train
    results = model.train(
        data=str(config_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device if args.device else None,
        workers=args.workers,
        name=run_name,
        project=str(project_root / "runs" / "detect"),
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        verbose=True,
        seed=42,
        deterministic=True,
        resume=args.resume,
        # Augmentation
        hsv_h=0.015,  # HSV-Hue augmentation
        hsv_s=0.7,    # HSV-Saturation augmentation
        hsv_v=0.4,    # HSV-Value augmentation
        degrees=10.0,  # Rotation (+/- deg)
        translate=0.1, # Translation (+/- fraction)
        scale=0.5,    # Scale (+/- gain)
        shear=2.0,    # Shear (+/- deg)
        flipud=0.5,   # Flip up-down probability
        fliplr=0.5,   # Flip left-right probability
        mosaic=1.0,   # Mosaic augmentation
        mixup=0.1,    # Mixup augmentation
    )

    # Results summary
    print(f"\n[3/3] Training complete!")
    print("=" * 60)

    # Find best model
    best_model_path = project_root / "runs" / "detect" / run_name / "weights" / "best.pt"

    if best_model_path.exists():
        print(f"\nBest model saved to: {best_model_path}")

        # Copy to models directory
        output_path = project_root / "models" / "yolo_leak_detector.pt"
        import shutil
        shutil.copy2(best_model_path, output_path)
        print(f"Copied to: {output_path}")

        print("\nNext steps:")
        print("  1. Evaluate: python src/evaluate_yolo.py")
        print("  2. Export to ONNX: yolo export model=models/yolo_leak_detector.pt format=onnx")
        print("  3. Run inference: python src/inference_yolo.py --image path/to/image.jpg")
    else:
        print("\nWARNING: Best model not found. Check training logs for errors.")

if __name__ == "__main__":
    main()
