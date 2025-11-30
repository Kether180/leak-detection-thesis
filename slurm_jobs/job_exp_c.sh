#!/bin/bash
#SBATCH --job-name=exp_c
#SBATCH --output=exp_c_%j.log
#SBATCH --partition=scavenge
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=6:00:00

source ~/.bashrc
conda activate sd_env
python train_resnet50.py --data_dir datasets/experiment_c --output model_exp_c.pth --epochs 35
