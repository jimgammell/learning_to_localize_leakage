#!/bin/bash

#SBATCH --job-name=ches-ctf-2018-highdropout-htune
#SBATCH --partition=cocosys
#SBATCH --account=cocosys
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=14
#SBATCH --time=4:00:00
#SBATCH --output=./outputs/ches_ctf_2018/htune_highdropout/slurm_%A_%a.out
#SBATCH --error=./outputs/ches_ctf_2018/htune_highdropout/slurm_%A_%a.out
#SBATCH --array=0-127%6

STRONG_ATTACKER_CKPT=./outputs/ches_ctf_2018/strong_attacker/seed_0/best_val_rank.ckpt

source ~/.bashrc
micromamba activate leakage-localization
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
bash ./bash_scripts/sup_train_and_eval.sh \
    ches_ctf_2018 \
    ./outputs/ches_ctf_2018/htune_highdropout/trial_${SLURM_ARRAY_TASK_ID} \
    $STRONG_ATTACKER_CKPT \
    --optuna-study-path ./outputs/ches_ctf_2018/htune_highdropout/study.log \
    --optuna-run-count 1 \
    --optuna-sampler qmc \
    --optuna-total-trials 128