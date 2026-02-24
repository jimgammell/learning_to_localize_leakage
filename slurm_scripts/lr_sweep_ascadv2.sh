#!/bin/bash

#SBATCH --job-name=ascadv2-lr-sweep
#SBATCH --partition=cocosys
#SBATCH --account=cocosys
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=14
#SBATCH --time=24:00:00
#SBATCH --output=./outputs/ascadv2/lr_sweep/slurm_%A_%a.out
#SBATCH --error=./outputs/ascadv2/lr_sweep/slurm_%A_%a.out
#SBATCH --array=0-3

LR_VALUES=(1e-5 3e-5 1e-4 3e-4)
LR=${LR_VALUES[$SLURM_ARRAY_TASK_ID]}

source ~/.bashrc
micromamba activate leakage-localization
python -m experiments.train.supervised \
    --dest ./outputs/ascadv2/lr_sweep/lr_${LR} \
    --config-file ascadv2 \
    --training.base_lr ${LR}