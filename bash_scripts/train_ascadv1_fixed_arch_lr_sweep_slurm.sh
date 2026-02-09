#!/bin/bash

#SBATCH --job-name=arch-lr-sweep
#SBATCH --partition=cocosys
#SBATCH --account=cocosys
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=14
#SBATCH --time=04:00:00
#SBATCH --output=arch-lr-sweep-%a.out
#SBATCH --error=arch-lr-sweep-%a.out
#SBATCH --array=0-35%6

source ~/.bashrc
micromamba activate leakage-localization

# ── Fixed regularization (best from previous sweep) ──────────────────
ROLL=4
HIDDEN_DROPOUT=0.3        # Best from previous: hdrop_0.3 was 2nd best
INPUT_DROPOUT=0.0         # Best from previous: idrop_0.0 was best
LABEL_SMOOTHING=0.0
WEIGHT_DECAY="1.e-4"
NOISE=0.0
POOLING="token"

# ── Grid: 6 architectures × 6 learning rates = 36 experiments ────────

# Learning rates to test
LR_VALUES=(1.e-4 2.e-4 3.e-4 4.e-4 5.e-4 6.e-4)

# Architecture combinations (position_embedding, fnn_style)
ARCH_CONFIGS=(
    "sinusoidal mlp"
    "sinusoidal gated"
    "rope mlp"
    "rope gated"
    "learned mlp"
    "learned gated"
)

# Calculate which architecture and LR from array index
ARCH_IDX=$((SLURM_ARRAY_TASK_ID / 6))
LR_IDX=$((SLURM_ARRAY_TASK_ID % 6))

# Extract architecture config
IFS=' ' read -r POSITION_EMBEDDING FNN_STYLE <<< "${ARCH_CONFIGS[$ARCH_IDX]}"

# Extract learning rate
BASE_LR="${LR_VALUES[$LR_IDX]}"

# Create descriptive name
NAME="${POSITION_EMBEDDING}_${FNN_STYLE}_lr${BASE_LR}"

echo "=== Experiment $SLURM_ARRAY_TASK_ID: $NAME ==="
echo "Architecture: $POSITION_EMBEDDING + $FNN_STYLE"
echo "Learning rate: $BASE_LR"

python -m experiments.train.supervised \
    --dest=./outputs/ascadv1_fixed/arch_lr_sweep/${NAME} \
    --config-file=ascadv1_fixed_transformer \
    --training.base_lr $BASE_LR \
    --data.random_roll $ROLL \
    --model.hidden_dropout_rate $HIDDEN_DROPOUT \
    --model.input_dropout_rate $INPUT_DROPOUT \
    --model.input_droppatch_rate 0.0 \
    --training.label_smoothing $LABEL_SMOOTHING \
    --training.weight_decay $WEIGHT_DECAY \
    --data.additive_gaussian_noise $NOISE \
    --model.position_embedding $POSITION_EMBEDDING \
    --model.fnn_style $FNN_STYLE \
    --model.pooling $POOLING