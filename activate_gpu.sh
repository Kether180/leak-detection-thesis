#!/bin/bash
echo "🚀 Requesting GPU node..."

srun -p scavenge \
     --gres=gpu:gtx1080ti:1 \
     --cpus-per-task=6 \
     --mem=30G \
     --time=02:00:00 \
     --pty bash --init-file <(echo "
module purge
module load Anaconda3/2024.02-1
source /opt/itu/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate sd_env
cd $REPO_DIR

echo ''
echo '=========================================='
echo '✅ GPU Session Ready!'
echo 'Node: \$(hostname)
echo 'Environment: \$CONDA_DEFAULT_ENV'
echo 'Working dir: \$(pwd)'
echo '=========================================='
echo ''

nvidia-smi
echo ''
")
