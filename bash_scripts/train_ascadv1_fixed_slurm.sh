#!/bin/bash

#SBATCH --job-name=train-ascadv1f-perceiver
#SBATCH --partition=cocosys
#SBATCH --account=cocosys
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=14
#SBATCH --time=24:00:00
#SBATCH --output=train-ascadv1f-perceiver.out
#SBATCH --error=train-ascadv1f-perceiver.out

source ~/.bashrc
micromamba activate leakage-localization
bash ./bash_scripts/train_ascadv1_fixed.sh