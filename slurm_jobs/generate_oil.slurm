#!/bin/bash
#SBATCH -p scavenge
#SBATCH --nodelist=cn16
#SBATCH --gres=gpu:gtx1080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH -o oil_synth_%j.log
#SBATCH -e oil_synth_%j.err

echo "[START] Oil synthetic job started at $(date)"

module load Anaconda3/2024.02-1
source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate sd_env

# REQUIRED — go to the folder where the script writes images
cd ${SLURM_SUBMIT_DIR}

# Debug visibility
echo "[CUDA] torch.cuda.is_available() = $(python - << EOF
import torch; print(torch.cuda.is_available())
EOF
)"

echo "[INFO] Generating OIL synthetic images..."
python data_utils/generate_synthetic_data.py --leak_type oil --num_images 390

echo "[DONE] Job finished at $(date)"



