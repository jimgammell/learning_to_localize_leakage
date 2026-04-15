#!/bin/bash
# Compute ta-mtd for missing baseline files:
#   - ascadv1_variable: oracle + random
#   - ches_ctf_2018:    random
# Run from the repo root: bash bash_scripts/eval_baselines_ta_mtd.sh

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python experiments/evaluate_trained_model.py \
    --path-to-eval ./outputs/ascadv1_variable/baselines/oracle.npy \
    --dataset ascadv1-variable \
    --metrics ta-mtd

python experiments/evaluate_trained_model.py \
    --path-to-eval ./outputs/ascadv1_variable/baselines/random.npy \
    --dataset ascadv1-variable \
    --metrics ta-mtd

python experiments/evaluate_trained_model.py \
    --path-to-eval ./outputs/ches_ctf_2018/baselines/random.npy \
    --dataset ches-ctf-2018 \
    --metrics ta-mtd