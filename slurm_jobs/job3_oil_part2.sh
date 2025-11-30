#!/bin/bash
#SBATCH -p scavenge
#SBATCH --gres=gpu:gtx1080ti:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=30G
#SBATCH --time=15:00:00
#SBATCH -o oil_part2_%j.log
#SBATCH -e oil_part2_%j.err
#SBATCH --job-name=gen_oil_p2

export PATH="/home/gega/.conda/envs/sd_env/bin:$PATH"
PYTHON="/home/gega/.conda/envs/sd_env/bin/python"
cd ${SLURM_SUBMIT_DIR}

echo "=== Finishing oil_leak Part 2 (resume) ==="
echo "Time: $(date)"
$PYTHON generate_synthetic_images.py --experiment b --class_name oil_leak --num_images 1400
echo "✓ oil_leak complete!"
echo "Time: $(date)"
