#!/bin/bash

#SBATCH --job-name=ascadv1f-strong-attacker
#SBATCH --partition=cocosys
#SBATCH --account=cocosys
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=14
#SBATCH --time=4:00:00
#SBATCH --output=./outputs/ascadv1_fixed/strong_attacker/slurm_%A_%a.out
#SBATCH --error=./outputs/ascadv1_fixed/strong_attacker/slurm_%A_%a.out
#SBATCH --array=0-4

source ~/.bashrc
micromamba activate leakage-localization
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
bash ./bash_scripts/sup_train_and_eval.sh \
    ascadv1_fixed \
    ./outputs/ascadv1_fixed/strong_attacker/seed_${SLURM_ARRAY_TASK_ID} \
    "" \
    --training.seed ${SLURM_ARRAY_TASK_ID}