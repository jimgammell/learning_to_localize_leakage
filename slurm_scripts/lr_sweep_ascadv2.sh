#!/bin/bash

#SBATCH --job-name=ascadv2-lr-sweep
#SBATCH --partition=cocosys
#SBATCH --account=cocosys
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=14
#SBATCH --time=48:00:00
#SBATCH --output=./outputs/ascadv2/lr_sweep_v2/slurm_%A_%a.out
#SBATCH --error=./outputs/ascadv2/lr_sweep_v2/slurm_%A_%a.out
#SBATCH --array=0-9

LR_VALUES=(1.e-5 3.e-5 1.e-4 3.e-4 1.e-3)
HEAD_VALUES=(ascadv2 null)

LR_IDX=$((SLURM_ARRAY_TASK_ID % 4))
HEAD_IDX=$((SLURM_ARRAY_TASK_ID / 4))

LR=${LR_VALUES[$LR_IDX]}
HEAD=${HEAD_VALUES[$HEAD_IDX]}

if [ "$HEAD" == "null" ]; then
    DEST=./outputs/ascadv2/lr_sweep_v2/no_head_lr_${LR}
else
    DEST=./outputs/ascadv2/lr_sweep_v2/ascadv2_head_lr_${LR}
fi

source ~/.bashrc
micromamba activate leakage-localization
python -m experiments.train.supervised \
    --dest ${DEST} \
    --config-file ascadv2 \
    --training.base_lr ${LR} \
    --model.grey_box_head ${HEAD}