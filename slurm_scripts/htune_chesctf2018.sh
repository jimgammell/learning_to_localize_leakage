#!/bin/bash

#SBATCH --job-name=chesctf2018-htune
#SBATCH --partition=cocosys
#SBATCH --account=cocosys
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=14
#SBATCH --time=12:00:00
#SBATCH --output=./outputs/ches_ctf_2018/reg_htune/slurm_%A_%a.out
#SBATCH --error=./outputs/ches_ctf_2018/reg_htune/slurm_%A_%a.out
#SBATCH --array=0-100%2

source ~/.bashrc
micromamba activate leakage-localization
python -m experiments.train.supervised \
    --dest ./outputs/ches_ctf_2018/reg_htune \
    --config-file ches_ctf_2018 \
    --optuna-study-path ./outputs/ches_ctf_2018/reg_htune/optuna_study.log \
    --sampler tpe