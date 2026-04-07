#!/bin/bash

#SBATCH --job-name=ascadv1v-highdropout-htune
#SBATCH --partition=cocosys
#SBATCH --account=cocosys
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=14
#SBATCH --time=8:00:00
#SBATCH --output=./outputs/ascadv1_variable/htune_highdropout/slurm_%A_%a.out
#SBATCH --error=./outputs/ascadv1_variable/htune_highdropout/slurm_%A_%a.out
#SBATCH --array=0-127%8

STRONG_ATTACKER_CKPT=./outputs/ascadv1_variable/strong_attacker/seed_0/best_val_rank.ckpt

RAMFS_ROOT=/dev/shm/ascadv1_variable_${SLURM_ARRAY_JOB_ID}
LOCK=/dev/shm/ascadv1_variable_${SLURM_ARRAY_JOB_ID}.lock
REFCOUNT=/dev/shm/ascadv1_variable_${SLURM_ARRAY_JOB_ID}.refcount

cleanup() {
    (
        flock -x 9
        count=$(cat "$REFCOUNT" 2>/dev/null || echo 1)
        new=$((count - 1))
        if [ "$new" -le 0 ]; then
            rm -rf "$RAMFS_ROOT" "$REFCOUNT" "$LOCK"
        else
            echo "$new" > "$REFCOUNT"
        fi
    ) 9>>"$LOCK"
}
trap cleanup EXIT

(
    flock -x 9
    count=$(cat "$REFCOUNT" 2>/dev/null || echo 0)
    echo $((count + 1)) > "$REFCOUNT"
    if [ ! -f "$RAMFS_ROOT/.ready" ]; then
        mkdir -p "$RAMFS_ROOT"
        find ./datasets/ascadv1_variable -maxdepth 1 -type f ! -name "*.h5" \
            -exec cp {} "$RAMFS_ROOT/" \;
        touch "$RAMFS_ROOT/.ready"
    fi
) 9>>"$LOCK"

source ~/.bashrc
micromamba activate leakage-localization
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
bash ./bash_scripts/sup_train_and_eval.sh \
    ascadv1_variable \
    ./outputs/ascadv1_variable/htune_highdropout/trial_${SLURM_ARRAY_TASK_ID} \
    $STRONG_ATTACKER_CKPT \
    --ascadv1-variable-root $RAMFS_ROOT \
    --optuna-study-path ./outputs/ascadv1_variable/htune_highdropout/study.log \
    --optuna-run-count 1 \
    --optuna-sampler qmc \
    --optuna-total-trials 128