#!/bin/bash
# Trains a model and caches all computationally-heavy evaluation results.
#
# Usage:
#   ./bash_scripts/sup_train_and_eval.sh CONFIG_FILE DEST [STRONG_ATTACKER_CKPT] [extra args...]
#
# Positional args:
#   CONFIG_FILE          - config file name (without .yaml extension)
#   DEST                 - output directory for this run's artifacts
#   STRONG_ATTACKER_CKPT - (optional) checkpoint of the canonical strong-attacker model
#                          (used for fwd/rev DNN occlusion tests); if omitted or empty,
#                          those two metrics are skipped
#
# Extra args are forwarded verbatim to train_supervised_model.py. Use this to
# pass Optuna args when running as a Slurm array job, e.g.:
#   --optuna-study-path ./outputs/.../study.log --optuna-run-count 1 --optuna-sampler qmc
#
# When --optuna-run-count 1 is passed, training artifacts land directly in DEST
# (no trial_N subdirectory), so the Slurm script should set DEST per array task:
#   DEST=./outputs/htune/trial_${SLURM_ARRAY_TASK_ID}

CONFIG_FILE=$1
DEST=$2
STRONG_ATTACKER_CKPT=$3
EXTRA_ARGS="${@:4}"

source ~/.bashrc
micromamba activate leakage-localization

python experiments/train_supervised_model.py \
    --config-file $CONFIG_FILE \
    --dest $DEST \
    $EXTRA_ARGS

# Compute attack performance for this run's model
python experiments/evaluate_trained_model.py \
    --model-ckpt-path $DEST/best_*.ckpt \
    --metrics attack-performance

# Compute attributions (loop so one crash doesn't block the other)
for attr_method in gradvis input-x-gradient
do
    python experiments/attribute_trained_model.py \
        --ckpt-path $DEST/best_*.ckpt \
        --attr-methods $attr_method
done

# Evaluate each attribution with each localization metric
for attr_method in gradvis input_x_gradient
do
    for eval_metric in white-box-agreement fwd-dnno-occl rev-dnno-occl ta-mtd
    do
        # Skip DNN occlusion tests if no strong-attacker checkpoint was provided
        if [[ "$eval_metric" == *"dnno-occl"* ]] && [[ -z "$STRONG_ATTACKER_CKPT" ]]; then
            continue
        fi
        ATTACKER_ARG=""
        if [[ -n "$STRONG_ATTACKER_CKPT" ]]; then
            ATTACKER_ARG="--strong-attacker-ckpt-path $STRONG_ATTACKER_CKPT"
        fi
        python experiments/evaluate_trained_model.py \
            --path-to-eval $DEST/$attr_method.npy \
            $ATTACKER_ARG \
            --metrics $eval_metric
    done
done