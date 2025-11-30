#!/bin/bash
#SBATCH -p scavenge
#SBATCH --gres=gpu:gtx1080ti:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=30G
#SBATCH --time=15:00:00
#SBATCH -o water_part2_%j.log
#SBATCH -e water_part2_%j.err
#SBATCH --job-name=gen_water_p2

export PATH="/home/gega/.conda/envs/sd_env/bin:$PATH"
PYTHON="/home/gega/.conda/envs/sd_env/bin/python"
cd ${SLURM_SUBMIT_DIR}

echo "=== Finishing water_leak Part 2 (resume) ==="
echo "Time: $(date)"
$PYTHON generate_synthetic_images.py --experiment b --class_name water_leak --num_images 1400
echo "✓ water_leak complete!"
echo "Time: $(date)"
