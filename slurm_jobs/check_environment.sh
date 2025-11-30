#!/bin/bash
# ============================================
# Environment Version Checker
# Run this on slurmhead (no GPU needed!)
# ============================================

echo "╔══════════════════════════════════════════════════════════╗"
echo "║          ENVIRONMENT VERSION CHECK                       ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Load Anaconda module
echo "→ Loading Anaconda module..."
module load Anaconda3/2024.02-1

# Activate environment
echo "→ Activating sd_env environment..."
source activate sd_env

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "                    SYSTEM INFO"
echo "═══════════════════════════════════════════════════════════"

# Python version
echo ""
echo "Python Version:"
python --version

# Python path
echo ""
echo "Python Path:"
which python

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "                    PYTORCH INFO"
echo "═══════════════════════════════════════════════════════════"

# PyTorch version
echo ""
python << 'EOF'
import torch
print(f"PyTorch Version:     {torch.__version__}")
print(f"CUDA Version:        {torch.version.cuda}")
print(f"cuDNN Version:       {torch.backends.cudnn.version()}")
print(f"CUDA Available:      {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device Count:   {torch.cuda.device_count()}")
EOF

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "                    KEY PACKAGES"
echo "═══════════════════════════════════════════════════════════"
echo ""

python << 'EOF'
import importlib.metadata

packages = [
    'torch',
    'torchvision',
    'diffusers',
    'transformers',
    'opencv-python',
    'pillow',
    'numpy',
    'pandas',
    'scikit-learn',
    'matplotlib',
    'tqdm'
]

print(f"{'Package':<20} {'Version':<15}")
print("-" * 35)

for pkg in packages:
    try:
        # Handle package name variations
        pkg_name = pkg
        if pkg == 'opencv-python':
            import cv2
            version = cv2.__version__
        elif pkg == 'pillow':
            import PIL
            version = PIL.__version__
        else:
            version = importlib.metadata.version(pkg)
        print(f"{pkg:<20} {version:<15}")
    except Exception as e:
        print(f"{pkg:<20} NOT FOUND")
EOF

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "                    SCRIPT VERIFICATION"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Check if key scripts exist
scripts=(
    "auto_filter_synthetic.py"
    "train_filter_classifier.py"
    "generate_synthetic_images.py"
    "src/train_resnet50.py"
    "generate_gradcam.py"
    "requirements.txt"
)

echo "Key Scripts:"
for script in "${scripts[@]}"; do
    if [ -f "$script" ]; then
        size=$(ls -lh "$script" | awk '{print $5}')
        echo "  ✓ $script ($size)"
    else
        echo "  ✗ $script (NOT FOUND)"
    fi
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "                    REQUIREMENTS.TXT CHECK"
echo "═══════════════════════════════════════════════════════════"
echo ""

if [ -f "requirements.txt" ]; then
    echo "Comparing installed vs requirements.txt:"
    echo ""
    
    # Check PyTorch version
    installed_torch=$(python -c "import torch; print(torch.__version__)")
    required_torch=$(grep "^torch==" requirements.txt | cut -d'=' -f3)
    
    if [ "$installed_torch" = "$required_torch" ]; then
        echo "  ✓ PyTorch: $installed_torch (matches requirements)"
    else
        echo "  ⚠ PyTorch: Installed=$installed_torch, Required=$required_torch"
    fi
    
    # Check Diffusers version
    installed_diff=$(python -c "import diffusers; print(diffusers.__version__)")
    required_diff=$(grep "^diffusers==" requirements.txt | cut -d'=' -f3)
    
    if [ "$installed_diff" = "$required_diff" ]; then
        echo "  ✓ Diffusers: $installed_diff (matches requirements)"
    else
        echo "  ⚠ Diffusers: Installed=$installed_diff, Required=$required_diff"
    fi
else
    echo "  ✗ requirements.txt not found"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "                    THESIS VERIFICATION"
echo "═══════════════════════════════════════════════════════════"
echo ""

python << 'EOF'
import torch

print("For your thesis, use these versions:")
print("")
print(f"  Python:   {'.'.join(map(str, __import__('sys').version_info[:2]))}")
print(f"  PyTorch:  {torch.__version__}")
print(f"  CUDA:     {torch.version.cuda}")
print("")
print("Update thesis if different from what you have written!")
EOF

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "                    COMPLETE ✓"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "No GPU needed for this check! ✓"
echo ""
