#!/bin/bash

#SBATCH --job-name=train-ascadv1f-perceiver
#SBATCH --partition=cocosys
#SBATCH --account=cocosys
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=14
#SBATCH --time=24:00:00
#SBATCH --output=train-ascadv1f-perceiver-%a.out
#SBATCH --error=train-ascadv1f-perceiver-%a.out
#SBATCH --array=0-7

LRS=(1.e-6 3.e-6 1.e-5 3.e-5 1.e-4 3.e-4 1.e-3 3.e-3)
LR=${LRS[$SLURM_ARRAY_TASK_ID]}

source ~/.bashrc
micromamba activate leakage-localization
python -m experiments.train.supervised \
    --dest=./outputs/ascadv1_fixed/supervised_lr${LR} \
    --config-file=ascadv1_fixed \
    --training.base_lr ${LR}