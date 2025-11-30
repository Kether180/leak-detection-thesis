"""
Run inference with trained YOLOv8 leak detection model.

Usage:
    python src/inference_yolo.py --image path/to/image.jpg
    python src/inference_yolo.py --image path/to/image.jpg --model models/yolo_leak_detector.pt
    python src/inference_yolo.py --folder path/to/images/
    python src/inference_yolo.py --video path/to/video.mp4
    python src/inference_yolo.py --webcam  # Live webcam detection
"""

import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Leak Detection Inference")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--folder", type=str, help="Path to folder of images")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--webcam", action="store_true", help="Use webcam")
    parser.add_argument(
        "--model",
        type=str,
        default="models/yolo_leak_detector.pt",
        help="Path to trained model"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to file"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display results"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Import YOLO
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed!")
        print("Run: pip install ultralytics")
        return

    # Check model exists
    project_root = Path(__file__).parent.parent
    model_path = project_root / args.model

    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("\nTrain a model first:")
        print("  1. Annotate images (Roboflow/Label Studio)")
        print("  2. Run: python src/train_yolo.py")
        return

    # Load model
    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    # Class names
    class_names = {0: "oil_leak", 1: "water_leak"}

    # Determine source
    if args.webcam:
        source = 0
        print("Starting webcam inference... Press 'q' to quit.")
    elif args.video:
        source = args.video
        print(f"Processing video: {source}")
    elif args.folder:
        source = args.folder
        print(f"Processing folder: {source}")
    elif args.image:
        source = args.image
        print(f"Processing image: {source}")
    else:
        print("ERROR: Please specify --image, --folder, --video, or --webcam")
        return

    # Run inference
    results = model.predict(
        source=source,
        conf=args.conf,
        iou=args.iou,
        save=args.save,
        show=args.show,
        stream=True if args.video or args.webcam else False,
    )

    # Process results
    print("\n" + "=" * 60)
    print("Detection Results")
    print("=" * 60)

    for i, result in enumerate(results):
        if result.boxes is not None and len(result.boxes) > 0:
            print(f"\nImage: {result.path}")
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                class_name = class_names.get(cls_id, f"class_{cls_id}")

                print(f"  - {class_name}: {conf:.2%} confidence")
                print(f"    Bbox: [{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}]")
        else:
            print(f"\nImage: {result.path}")
            print("  - No leaks detected")

    if args.save:
        print(f"\nResults saved to: runs/detect/predict/")

if __name__ == "__main__":
    main()
