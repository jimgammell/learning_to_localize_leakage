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

LR=(2.e-4 3.e-4 4.e-4 5.e-4 6.e-4 7.e-4 8.e-4 9.e-4)

IDX=$SLURM_ARRAY_TASK_ID

source ~/.bashrc
micromamba activate leakage-localization
python -m experiments.train.supervised \
    --dest=./outputs/ascadv1_fixed/perceiver_supervised_reg_lr${LR[$IDX]} \
    --config-file=ascadv1_fixed \
    --training.base_lr ${LR[$IDX]}