#!/bin/bash
#SBATCH -p scavenge
#SBATCH --gres=gpu:gtx1080ti:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=30G
#SBATCH --time=20:00:00
#SBATCH -o no_leak_%j.log
#SBATCH -e no_leak_%j.err
#SBATCH --job-name=gen_no_leak

export PATH="/home/gega/.conda/envs/sd_env/bin:$PATH"
PYTHON="/home/gega/.conda/envs/sd_env/bin/python"
cd ${SLURM_SUBMIT_DIR}

echo "=== Finishing no_leak images ==="
echo "Time: $(date)"
$PYTHON generate_synthetic_images.py --experiment b --class_name no_leak --num_images 1400
echo "✓ no_leak complete!"
echo "Time: $(date)"
