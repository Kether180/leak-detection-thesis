#!/bin/bash
#SBATCH --job-name=setup_env
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/setup_env_%j.out

# ===================================
#  Leak Detection Project - Setup Script
# ===================================

echo "------------------------------------------"
echo "   Starting Environment Setup on HPC"
echo "------------------------------------------"

# -------- 1. Load Anaconda module --------
echo "[1/6] Loading Anaconda module..."
module load Anaconda3/2024.02-1

# -------- 2. Create conda environment --------
ENV_NAME="torchenv"

echo "[2/6] Creating conda environment: $ENV_NAME"
conda create -y -n $ENV_NAME python=3.10

# -------- 3. Activate environment --------
echo "[3/6] Activating conda environment..."
source activate $ENV_NAME

# -------- 4. Install CUDA-enabled PyTorch --------
echo "[4/6] Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# -------- 5. Install project dependencies --------
echo "[5/6] Installing project requirements..."
pip install -r requirements.txt

# -------- 6. Validate environment --------
echo "[6/6] Verifying installation..."

echo -e "\n>>> Python version:"
python --version

echo -e "\n>>> PyTorch CUDA availability:"
python - <<EOF
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
EOF

echo "------------------------------------------"
echo " Setup Completed Successfully! "
echo "------------------------------------------"

