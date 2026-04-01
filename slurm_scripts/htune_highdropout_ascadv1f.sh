#!/bin/bash

#SBATCH --job-name=ascadv1f-highdropout-htune
#SBATCH --partition=cocosys
#SBATCH --account=cocosys
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=14
#SBATCH --time=4:00:00
#SBATCH --output=./outputs/ascadv1_fixed/htune_highdropout/slurm_%A_%a.out
#SBATCH --error=./outputs/ascadv1_fixed/htune_highdropout/slurm_%A_%a.out
#SBATCH --array=0-127%5

STRONG_ATTACKER_CKPT=./outputs/ascadv1_fixed/strong_attacker/seed_0/best_val_rank.ckpt

source ~/.bashrc
micromamba activate leakage-localization
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
bash ./bash_scripts/sup_train_and_eval.sh \
    ascadv1_fixed \
    ./outputs/ascadv1_fixed/htune_highdropout/trial_${SLURM_ARRAY_TASK_ID} \
    $STRONG_ATTACKER_CKPT \
    --optuna-study-path ./outputs/ascadv1_fixed/htune_highdropout/study.log \
    --optuna-run-count 1 \
    --optuna-sampler qmc \
    --optuna-total-trials 128