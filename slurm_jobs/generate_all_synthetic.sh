#!/bin/bash
#SBATCH -p scavenge
#SBATCH --gres=gpu:gtx1080ti:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=30G
#SBATCH --time=120:00:00
#SBATCH -o generation_full_%j.log
#SBATCH -e generation_full_%j.err
#SBATCH --job-name=gen_all_synthetic

echo "=========================================="
echo "Mass Synthetic Generation Started"
echo "Time: $(date)"
echo "=========================================="

# Set paths explicitly
export PATH="/home/gega/.conda/envs/sd_env/bin:$PATH"
PYTHON="/home/gega/.conda/envs/sd_env/bin/python"

cd ${SLURM_SUBMIT_DIR}

echo ""
echo "Python: $PYTHON"
echo "Testing torch import..."
$PYTHON -c "import torch; print('✅ PyTorch available')"
echo ""

echo "Generating 4,200 synthetic images (1,400 per class)"
echo "This will take ~15-18 hours"
echo ""

echo "1/3: Generating no_leak images..."
$PYTHON generate_synthetic_images.py --experiment b --class_name no_leak --num_images 1400

echo ""
echo "2/3: Generating oil_leak images..."
$PYTHON generate_synthetic_images.py --experiment b --class_name oil_leak --num_images 1400

echo ""
echo "3/3: Generating water_leak images..."
$PYTHON generate_synthetic_images.py --experiment b --class_name water_leak --num_images 1400

echo ""
echo "=========================================="
echo "Generation Complete!"
echo "Time: $(date)"
echo "Total images: 4,200"
echo "=========================================="
