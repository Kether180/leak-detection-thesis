#!/bin/bash
#SBATCH -p scavenge
#SBATCH --nodelist=cn16
#SBATCH --gres=gpu:gtx1080ti:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=30G
#SBATCH --time=12:00:00
#SBATCH -o train_%j.log
#SBATCH -e train_%j.err
#SBATCH --job-name=train_resnet50

echo "Training started at $(date)"

module purge
module load Anaconda3/2024.02-1
source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh

conda activate sd_env
export PATH=$CONDA_PREFIX/bin:$PATH

cd ${SLURM_SUBMIT_DIR}

python -u src/train_resnet50.py

echo "Training finished at $(date)"

