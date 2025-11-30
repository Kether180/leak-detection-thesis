#!/bin/bash
#SBATCH --job-name=supp_water
#SBATCH --output=supplement_water_%j.log
#SBATCH --error=supplement_water_%j.log
#SBATCH --partition=scavenge
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=6:00:00

echo "=== Generating Water-Leak Supplement (1400-1499) ==="
echo "Time: $(date)"
source ~/.bashrc
conda activate sd_env
python generate_supplement.py --class_name water_leak --start_idx 1400 --end_idx 1500
echo "✅ Complete! Time: $(date)"
