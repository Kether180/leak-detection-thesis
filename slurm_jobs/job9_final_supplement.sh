#!/bin/bash
#SBATCH --job-name=final_supp
#SBATCH --output=final_supplement_%j.log
#SBATCH --error=final_supplement_%j.log
#SBATCH --partition=scavenge
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=4:00:00

echo "=== Final Supplement to reach 4,500 ==="
echo "Time: $(date)"
source ~/.bashrc
conda activate sd_env

# Generate remaining for each class
python generate_supplement.py --class_name no_leak --start_idx 1481 --end_idx 1500
python generate_supplement.py --class_name oil_leak --start_idx 1480 --end_idx 1500
python generate_supplement.py --class_name water_leak --start_idx 1482 --end_idx 1500

echo "✅ Complete! Time: $(date)"
