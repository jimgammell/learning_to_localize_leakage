#!/bin/bash

#SBATCH --job-name=rope-lr-seed
#SBATCH --partition=cocosys
#SBATCH --account=cocosys
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=14
#SBATCH --time=04:00:00
#SBATCH --output=rope-lr-seed-%a.out
#SBATCH --error=rope-lr-seed-%a.out
#SBATCH --array=0-35%6

source ~/.bashrc
micromamba activate leakage-localization

# ── Fixed config (best from previous sweeps) ─────────────────────────
ROLL=4
HIDDEN_DROPOUT=0.3
INPUT_DROPOUT=0.0
LABEL_SMOOTHING=0.0
WEIGHT_DECAY="1.e-4"
NOISE=0.0
POOLING="token"
POSITION_EMBEDDING="rope"
FNN_STYLE="mlp"

# ── Grid: 12 learning rates × 3 seeds = 36 experiments ───────────────

LR_VALUES=(8.e-5 9.e-5 1.e-4 2.e-4 3.e-4 4.e-4 5.e-4 6.e-4 7.e-4 8.e-4 9.e-4 1.e-3)
SEEDS=(0 1 2)

LR_IDX=$((SLURM_ARRAY_TASK_ID / 3))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 3))

BASE_LR="${LR_VALUES[$LR_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

NAME="rope_mlp_lr${BASE_LR}_seed${SEED}"

echo "=== Experiment $SLURM_ARRAY_TASK_ID: $NAME ==="

python -m experiments.train.supervised \
    --dest=./outputs/ascadv1_fixed/rope_lr_seed_sweep/${NAME} \
    --config-file=ascadv1_fixed_transformer \
    --training.base_lr $BASE_LR \
    --training.seed $SEED \
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