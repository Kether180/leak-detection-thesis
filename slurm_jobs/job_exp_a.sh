#!/bin/bash
#SBATCH --job-name=exp_a
#SBATCH --output=exp_a_%j.log
#SBATCH --partition=scavenge
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=6:00:00

source ~/.bashrc
conda activate sd_env
python train_resnet50.py --data_dir datasets/experiment_a --output model_exp_a.pth --epochs 35
