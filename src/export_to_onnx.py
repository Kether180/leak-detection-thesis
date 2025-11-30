"""
Export trained ResNet50 leak detection model to ONNX format
- ResNet50 architecture
- 3 classes: oil_leak, water_leak, no_leak
- ImageNet pretrained weights (IMAGENET1K_V2)
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights
import onnx
import onnxruntime as ort

# Paths
MODEL_PATH = "models/model_exp_c.pth"  # Hybrid model (best: 99.37% accuracy)
OUTPUT_PATH = "models/onnx/leak_detector_resnet50.onnx"

print("=" * 60)
print("ONNX Export for Leak Detection Model")
print("=" * 60)

# Load model architecture
device = torch.device("cpu")  # Force CPU for safer ONNX export
weights = ResNet50_Weights.IMAGENET1K_V2
model = models.resnet50(weights=weights)

# Replace final layer for 3-class classification
num_classes = 3  # oil_leak, water_leak, no_leak
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Load trained weights
print(f"\n[1/4] Loading trained model from: {MODEL_PATH}")
checkpoint = torch.load(MODEL_PATH, map_location="cpu")

# Handle both formats: dict with 'model_state_dict' or raw state_dict
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"   Validation accuracy: {checkpoint.get('val_accuracy', 'N/A')}")
else:
    model.load_state_dict(checkpoint)

model.eval()
model.cpu()
print("✓ Model loaded successfully")

# Create dummy input (batch_size=1, channels=3, height=224, width=224)
dummy_input = torch.randn(1, 3, 224, 224, device="cpu")

# Export to ONNX
print(f"\n[2/4] Exporting to ONNX format...")
torch.onnx.export(
    model,
    dummy_input,
    OUTPUT_PATH,
    input_names=["input"],
    output_names=["output"],
    opset_version=17,
    dynamic_axes={
        "input": {0: "batch_size"},   # Variable batch size
        "output": {0: "batch_size"}
    },
    export_params=True,
    do_constant_folding=True,  # Optimize by folding constant operations
    verbose=False
)
print(f"✓ Exported ONNX model to: {OUTPUT_PATH}")

# Validate ONNX model structure
print(f"\n[3/4] Validating ONNX model structure...")
onnx_model = onnx.load(OUTPUT_PATH)
onnx.checker.check_model(onnx_model)
print("✓ ONNX model structure is valid")

# Test inference with ONNX Runtime
print(f"\n[4/4] Testing inference with ONNX Runtime...")
session = ort.InferenceSession(OUTPUT_PATH, providers=["CPUExecutionProvider"])

# Get input/output info
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print(f"   Input: {input_name}, shape: {session.get_inputs()[0].shape}")
print(f"   Output: {output_name}, shape: {session.get_outputs()[0].shape}")

# Run test inference
test_input = dummy_input.numpy()
outputs = session.run([output_name], {input_name: test_input})
predictions = outputs[0]

print(f"\n✓ ONNX Runtime validation successful!")
print(f"   Test output shape: {predictions.shape}")
print(f"   Test predictions (logits): {predictions[0]}")

# Model info
import os
model_size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
print(f"\n{'=' * 60}")
print(f"ONNX Export Complete!")
print(f"{'=' * 60}")
print(f"Model file: {OUTPUT_PATH}")
print(f"Model size: {model_size_mb:.2f} MB")
print(f"Architecture: ResNet50")
print(f"Number of classes: 3 (oil_leak, water_leak, no_leak)")
print(f"Input shape: (batch_size, 3, 224, 224)")
print(f"Output shape: (batch_size, 3)")
print(f"\nReady for deployment on:")
print(f"  - NVIDIA Jetson devices (Orin, Xavier)")
print(f"  - Intel/AMD CPUs")
print(f"  - ARM processors (Raspberry Pi)")
print(f"  - Cloud platforms (AWS, Azure, GCP)")
print(f"  - ROS2 robotic systems")
print(f"{'=' * 60}")
