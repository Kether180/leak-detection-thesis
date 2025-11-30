"""
Export trained YOLOv8 leak detection model to ONNX format.

This enables deployment on:
- NVIDIA Jetson (with TensorRT optimization)
- Raspberry Pi (ONNX Runtime)
- Intel/AMD CPUs
- Cloud platforms
- ROS2 robotic systems

Usage:
    python src/export_yolo_to_onnx.py
    python src/export_yolo_to_onnx.py --model models/yolo_leak_detector.pt
    python src/export_yolo_to_onnx.py --simplify  # Simplify ONNX graph
    python src/export_yolo_to_onnx.py --half      # FP16 (for GPU inference)
"""

import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLOv8 to ONNX")
    parser.add_argument(
        "--model",
        type=str,
        default="models/yolo_leak_detector.pt",
        help="Path to trained YOLOv8 model"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size"
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Simplify ONNX graph"
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Export as FP16 (half precision)"
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic batch size"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version"
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

    # Paths
    project_root = Path(__file__).parent.parent
    model_path = project_root / args.model

    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("\nTrain a model first: python src/train_yolo.py")
        return

    print("=" * 60)
    print("YOLOv8 ONNX Export for Leak Detection")
    print("=" * 60)

    # Load model
    print(f"\n[1/3] Loading model: {model_path}")
    model = YOLO(str(model_path))

    # Export to ONNX
    print(f"\n[2/3] Exporting to ONNX...")
    print(f"  - Image size: {args.imgsz}")
    print(f"  - Opset: {args.opset}")
    print(f"  - Simplify: {args.simplify}")
    print(f"  - Half (FP16): {args.half}")
    print(f"  - Dynamic batch: {args.dynamic}")

    output_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        simplify=args.simplify,
        half=args.half,
        dynamic=args.dynamic,
        opset=args.opset,
    )

    print(f"\n[3/3] Export complete!")
    print("=" * 60)

    # Move to models/onnx directory
    onnx_output = project_root / "models" / "onnx" / "yolo_leak_detector.onnx"
    onnx_output.parent.mkdir(parents=True, exist_ok=True)

    if Path(output_path).exists():
        import shutil
        shutil.move(output_path, onnx_output)
        print(f"\nONNX model saved to: {onnx_output}")

        # Get file size
        size_mb = onnx_output.stat().st_size / (1024 * 1024)
        print(f"Model size: {size_mb:.2f} MB")

        print("\nDeployment options:")
        print("  - Jetson: Convert to TensorRT for 2-3x speedup")
        print("    trtexec --onnx=models/onnx/yolo_leak_detector.onnx --saveEngine=yolo.trt")
        print("  - CPU: Use ONNX Runtime directly")
        print("  - ROS2: Load ONNX in detection node")

        print("\nTest inference:")
        print("  python -c \"")
        print("  import onnxruntime as ort")
        print("  session = ort.InferenceSession('models/onnx/yolo_leak_detector.onnx')")
        print("  print('Model loaded successfully!')\"")

if __name__ == "__main__":
    main()
