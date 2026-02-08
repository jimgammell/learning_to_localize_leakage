#!/bin/bash

#SBATCH --job-name=train-ascadv1f-transformer
#SBATCH --partition=cocosys
#SBATCH --account=cocosys
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=14
#SBATCH --time=24:00:00
#SBATCH --output=train-ascadv1f-transformer-%a.out
#SBATCH --error=train-ascadv1f-transformer-%a.out
#SBATCH --array=0-7

ROLL=(0 1 2 4 8 16 32 64)

IDX=$SLURM_ARRAY_TASK_ID

source ~/.bashrc
micromamba activate leakage-localization
python -m experiments.train.supervised \
    --dest=./outputs/ascadv1_fixed/transformer_supervised_reg_roll${ROLL[$IDX]} \
    --config-file=ascadv1_fixed_transformer \
    --training.base_lr 2.e-4 \
    --data.random_roll ${ROLL[$IDX]}