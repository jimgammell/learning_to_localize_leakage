#!/bin/bash

#SBATCH --job-name=ascadv1v-reg-sweep
#SBATCH --partition=cocosys
#SBATCH --account=cocosys
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=14
#SBATCH --time=8:00:00
#SBATCH --output=./outputs/ascadv1_variable/reg_sweep/slurm_%A_%a.out
#SBATCH --error=./outputs/ascadv1_variable/reg_sweep/slurm_%A_%a.out
#SBATCH --array=0-146%2

# 7 sweeps x 3 seeds = total array tasks
# Total configs: 10+10+5+6+6+6+6 = 49
# Total tasks: 49 * 3 seeds = 147 (indices 0-146)
# All non-swept regularizers stay at yaml defaults.

source ~/.bashrc
micromamba activate leakage-localization

TASK_ID=$SLURM_ARRAY_TASK_ID
SEED=$((TASK_ID % 3))
CONFIG_ID=$((TASK_ID / 3))

INPUT_DROPOUT=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
HIDDEN_DROPOUT=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
WEIGHT_DECAY=(1.0e-4 1.0e-3 1.0e-2 1.0e-1 1.0e0)
GAUSSIAN_NOISE=(0. 1.0e-4 1.0e-3 1.0e-2 1.0e-1 1.0e0)
RANDOM_ROLL=(0. 1. 2. 4. 8. 16.)
RANDOM_LPF=(0. 1. 2. 4. 8. 16.)
MIXUP=(0. 0.2 0.4 0.6 0.8 1.0)

OFFSET=0
BASE=./outputs/ascadv1_variable/reg_sweep

# Input dropout sweep (configs 0-9)
if [ $CONFIG_ID -lt $((OFFSET + ${#INPUT_DROPOUT[@]})) ]; then
    IDX=$((CONFIG_ID - OFFSET))
    VAL=${INPUT_DROPOUT[$IDX]}
    python -m experiments.train.supervised \
        --dest ${BASE}/input_dropout_${VAL}/seed_${SEED} \
        --config-file ascadv1_variable \
        --model.input_dropout_rate ${VAL} \
        --training.seed ${SEED}
    exit $?
fi
OFFSET=$((OFFSET + ${#INPUT_DROPOUT[@]}))

# Hidden dropout sweep (configs 10-19)
if [ $CONFIG_ID -lt $((OFFSET + ${#HIDDEN_DROPOUT[@]})) ]; then
    IDX=$((CONFIG_ID - OFFSET))
    VAL=${HIDDEN_DROPOUT[$IDX]}
    python -m experiments.train.supervised \
        --dest ${BASE}/hidden_dropout_${VAL}/seed_${SEED} \
        --config-file ascadv1_variable \
        --model.hidden_dropout_rate ${VAL} \
        --training.seed ${SEED}
    exit $?
fi
OFFSET=$((OFFSET + ${#HIDDEN_DROPOUT[@]}))

# Weight decay sweep (configs 20-24)
if [ $CONFIG_ID -lt $((OFFSET + ${#WEIGHT_DECAY[@]})) ]; then
    IDX=$((CONFIG_ID - OFFSET))
    VAL=${WEIGHT_DECAY[$IDX]}
    python -m experiments.train.supervised \
        --dest ${BASE}/weight_decay_${VAL}/seed_${SEED} \
        --config-file ascadv1_variable \
        --training.weight_decay ${VAL} \
        --training.seed ${SEED}
    exit $?
fi
OFFSET=$((OFFSET + ${#WEIGHT_DECAY[@]}))

# Gaussian noise sweep (configs 25-30)
if [ $CONFIG_ID -lt $((OFFSET + ${#GAUSSIAN_NOISE[@]})) ]; then
    IDX=$((CONFIG_ID - OFFSET))
    VAL=${GAUSSIAN_NOISE[$IDX]}
    python -m experiments.train.supervised \
        --dest ${BASE}/gaussian_noise_${VAL}/seed_${SEED} \
        --config-file ascadv1_variable \
        --training.additive_gaussian_noise ${VAL} \
        --training.seed ${SEED}
    exit $?
fi
OFFSET=$((OFFSET + ${#GAUSSIAN_NOISE[@]}))

# Random roll sweep (configs 31-36)
if [ $CONFIG_ID -lt $((OFFSET + ${#RANDOM_ROLL[@]})) ]; then
    IDX=$((CONFIG_ID - OFFSET))
    VAL=${RANDOM_ROLL[$IDX]}
    python -m experiments.train.supervised \
        --dest ${BASE}/random_roll_${VAL}/seed_${SEED} \
        --config-file ascadv1_variable \
        --data.random_roll_scale ${VAL} \
        --training.seed ${SEED}
    exit $?
fi
OFFSET=$((OFFSET + ${#RANDOM_ROLL[@]}))

# Random LPF sweep (configs 37-42)
if [ $CONFIG_ID -lt $((OFFSET + ${#RANDOM_LPF[@]})) ]; then
    IDX=$((CONFIG_ID - OFFSET))
    VAL=${RANDOM_LPF[$IDX]}
    python -m experiments.train.supervised \
        --dest ${BASE}/random_lpf_${VAL}/seed_${SEED} \
        --config-file ascadv1_variable \
        --data.random_lpf_scale ${VAL} \
        --training.seed ${SEED}
    exit $?
fi
OFFSET=$((OFFSET + ${#RANDOM_LPF[@]}))

# Mixup sweep (configs 43-48)
if [ $CONFIG_ID -lt $((OFFSET + ${#MIXUP[@]})) ]; then
    IDX=$((CONFIG_ID - OFFSET))
    VAL=${MIXUP[$IDX]}
    python -m experiments.train.supervised \
        --dest ${BASE}/mixup_${VAL}/seed_${SEED} \
        --config-file ascadv1_variable \
        --training.mixup_alpha ${VAL} \
        --training.seed ${SEED}
    exit $?
fi
OFFSET=$((OFFSET + ${#MIXUP[@]}))

echo "ERROR: CONFIG_ID=${CONFIG_ID} out of range"
exit 1