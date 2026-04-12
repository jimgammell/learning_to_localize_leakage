#!/bin/bash
# Computes ta-mtd metrics for any trial in htune_highdropout that is missing them.
# Safe to re-run: skips trials where the output .npz already exists.
#
# Usage:
#   ./bash_scripts/compute_missing_ta_mtd.sh

source ~/.bashrc
micromamba activate leakage-localization

BASE=./outputs/ascadv1_variable/htune_highdropout

for trial_dir in $BASE/trial_*/; do
    for attr_method in gradvis input_x_gradient; do
        out="$trial_dir/ta_mtd.${attr_method}.npz"
        attr_file="$trial_dir/${attr_method}.npy"
        if [[ ! -f "$out" ]] && [[ -f "$attr_file" ]]; then
            echo "Computing ta_mtd for $trial_dir / $attr_method"
            python experiments/evaluate_trained_model.py \
                --path-to-eval "$attr_file" \
                --metrics ta-mtd
        fi
    done
done