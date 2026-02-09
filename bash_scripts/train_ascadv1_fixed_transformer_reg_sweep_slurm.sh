#!/bin/bash

#SBATCH --job-name=reg-sweep
#SBATCH --partition=cocosys
#SBATCH --account=cocosys
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=14
#SBATCH --time=04:00:00
#SBATCH --output=reg-sweep-%a.out
#SBATCH --error=reg-sweep-%a.out
#SBATCH --array=0-43%4

source ~/.bashrc
micromamba activate leakage-localization

# ── Defaults (= baseline) ──────────────────────────────────────────
BASE_LR="2.e-4"
ROLL=4
HIDDEN_DROPOUT=0.1
INPUT_DROPOUT=0.1
LABEL_SMOOTHING=0.0
WEIGHT_DECAY="1.e-4"
NOISE=0.0
POSITION_EMBEDDING="sinusoidal"
FNN_STYLE="mlp"
POOLING="token"
ACCUM=1

# ── Per-experiment overrides ────────────────────────────────────────
case $SLURM_ARRAY_TASK_ID in

    # === Baseline ===
    0) NAME="baseline" ;;

    # === Architecture tweaks (single-factor) ===
    1) NAME="arch_rope";              POSITION_EMBEDDING="rope" ;;
    2) NAME="arch_learned";           POSITION_EMBEDDING="learned" ;;
    3) NAME="arch_gated";             FNN_STYLE="gated" ;;
    4) NAME="arch_rope_gated";        POSITION_EMBEDDING="rope"; FNN_STYLE="gated" ;;

    # === Hidden dropout sweep ===
    5)  NAME="hdrop_0.0";   HIDDEN_DROPOUT=0.0 ;;
    6)  NAME="hdrop_0.05";  HIDDEN_DROPOUT=0.05 ;;
    7)  NAME="hdrop_0.2";   HIDDEN_DROPOUT=0.2 ;;
    8)  NAME="hdrop_0.3";   HIDDEN_DROPOUT=0.3 ;;

    # === Input dropout sweep ===
    9)  NAME="idrop_0.0";   INPUT_DROPOUT=0.0 ;;
    10) NAME="idrop_0.2";   INPUT_DROPOUT=0.2 ;;
    11) NAME="idrop_0.3";   INPUT_DROPOUT=0.3 ;;

    # === Weight decay sweep ===
    12) NAME="wd_0";     WEIGHT_DECAY="0." ;;
    13) NAME="wd_1e-3";  WEIGHT_DECAY="1.e-3" ;;
    14) NAME="wd_1e-2";  WEIGHT_DECAY="1.e-2" ;;

    # === Gaussian noise sweep ===
    15) NAME="noise_0.1";   NOISE=0.1 ;;
    16) NAME="noise_0.25";  NOISE=0.25 ;;
    17) NAME="noise_0.5";   NOISE=0.5 ;;

    # === Roll check ===
    18) NAME="roll_32";   ROLL=32 ;;
    19) NAME="roll_128";  ROLL=128 ;;

    # === Combined regularization ===
    20) NAME="combo_heavy_drop";
        HIDDEN_DROPOUT=0.2; INPUT_DROPOUT=0.2 ;;
    21) NAME="combo_nodrop";
        HIDDEN_DROPOUT=0.0; INPUT_DROPOUT=0.0 ;;
    22) NAME="combo_noise_hdrop";
        NOISE=0.25; HIDDEN_DROPOUT=0.2; INPUT_DROPOUT=0.2 ;;
    23) NAME="combo_noise_moddrop";
        NOISE=0.1; HIDDEN_DROPOUT=0.15; INPUT_DROPOUT=0.15 ;;
    24) NAME="combo_nodrop_noise";
        HIDDEN_DROPOUT=0.0; INPUT_DROPOUT=0.0; NOISE=0.25 ;;
    25) NAME="combo_wd_moddrop";
        WEIGHT_DECAY="1.e-3"; HIDDEN_DROPOUT=0.15; INPUT_DROPOUT=0.15 ;;
    26) NAME="combo_noise_wd";
        NOISE=0.25; WEIGHT_DECAY="1.e-3" ;;
    27) NAME="combo_mild_all";
        HIDDEN_DROPOUT=0.15; INPUT_DROPOUT=0.15; NOISE=0.1; WEIGHT_DECAY="1.e-3" ;;
    28) NAME="combo_strong_all";
        HIDDEN_DROPOUT=0.2; INPUT_DROPOUT=0.2; NOISE=0.25; WEIGHT_DECAY="1.e-3" ;;

    # === Architecture + regularization combos ===
    29) NAME="arch_rope_gated_hdrop";
        POSITION_EMBEDDING="rope"; FNN_STYLE="gated"; HIDDEN_DROPOUT=0.2; INPUT_DROPOUT=0.2 ;;
    30) NAME="arch_rope_gated_noise";
        POSITION_EMBEDDING="rope"; FNN_STYLE="gated"; NOISE=0.25 ;;
    31) NAME="arch_rope_gated_wd";
        POSITION_EMBEDDING="rope"; FNN_STYLE="gated"; WEIGHT_DECAY="1.e-3" ;;
    32) NAME="arch_rope_gated_mild";
        POSITION_EMBEDDING="rope"; FNN_STYLE="gated"; HIDDEN_DROPOUT=0.15; INPUT_DROPOUT=0.15; NOISE=0.1; WEIGHT_DECAY="1.e-3" ;;
    33) NAME="arch_rope_gated_strong";
        POSITION_EMBEDDING="rope"; FNN_STYLE="gated"; HIDDEN_DROPOUT=0.2; INPUT_DROPOUT=0.2; NOISE=0.25; WEIGHT_DECAY="1.e-3" ;;

    # === LR interaction ===
    34) NAME="lr_high_morereg";
        BASE_LR="4.e-4"; HIDDEN_DROPOUT=0.2; INPUT_DROPOUT=0.2 ;;
    35) NAME="lr_low_lessreg";
        BASE_LR="1.e-4"; HIDDEN_DROPOUT=0.05; INPUT_DROPOUT=0.05 ;;
    36) NAME="lr_high_rope_gated";
        BASE_LR="4.e-4"; POSITION_EMBEDDING="rope"; FNN_STYLE="gated" ;;

    # === Label smoothing (limited — may hurt) ===
    37) NAME="ls_0.05";             LABEL_SMOOTHING=0.05 ;;
    38) NAME="ls_0.1";              LABEL_SMOOTHING=0.1 ;;
    39) NAME="ls_0.1_rope_gated";   LABEL_SMOOTHING=0.1; POSITION_EMBEDDING="rope"; FNN_STYLE="gated" ;;

    # === Attention pooling (limited — may hurt) ===
    40) NAME="attnpool";
        POOLING="attention" ;;
    41) NAME="attnpool_rope_gated";
        POOLING="attention"; POSITION_EMBEDDING="rope"; FNN_STYLE="gated" ;;

    # === Misc combos ===
    42) NAME="arch_rope_gated_roll128";
        POSITION_EMBEDDING="rope"; FNN_STYLE="gated"; ROLL=128 ;;
    43) NAME="combo_nodrop_noise_wd";
        HIDDEN_DROPOUT=0.0; INPUT_DROPOUT=0.0; NOISE=0.25; WEIGHT_DECAY="1.e-3" ;;

esac

echo "=== Experiment $SLURM_ARRAY_TASK_ID: $NAME ==="

python -m experiments.train.supervised \
    --dest=./outputs/ascadv1_fixed/transformer_reg_sweep/${NAME} \
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
    --model.pooling $POOLING \
    --training.accumulate_grad_batches $ACCUM