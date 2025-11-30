#!/bin/bash
#SBATCH -p scavenge
#SBATCH --nodelist=cn16
#SBATCH --gres=gpu:gtx1080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH -o leak_synth_%j.log
#SBATCH -e leak_synth_%j.err

echo "[START] Job started at $(date)"

# Load conda properly inside SLURM
module load Anaconda3/2024.02-1
source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate sd_env

cd ${SLURM_SUBMIT_DIR}

echo "[INFO] Generating WATER images..."
python data_utils/generate_synthetic_data.py --leak_type water --num_images 241

echo "[INFO] Generating OIL images..."
python data_utils/generate_synthetic_data.py --leak_type oil --num_images 237

echo "[DONE] Job finished at $(date)"

