#!/bin/bash
#
# Evaluate fwd-dnno-occl + rev-dnno-occl for the three missing baseline files:
#   0: ascadv1_variable / oracle
#   1: ascadv1_variable / random
#   2: ches_ctf_2018   / random
#
# Submit with:
#   sbatch slurm_scripts/eval_baselines.sh

#SBATCH --job-name=eval-baselines
#SBATCH --partition=cocosys
#SBATCH --account=cocosys
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=14
#SBATCH --time=4:00:00
#SBATCH --output=./outputs/slurm_eval_baselines_%A_%a.out
#SBATCH --error=./outputs/slurm_eval_baselines_%A_%a.out
#SBATCH --array=0-2

case $SLURM_ARRAY_TASK_ID in
  0)
    DATASET=ascadv1_variable
    PATH_TO_EVAL=./outputs/ascadv1_variable/baselines/oracle.npy
    STRONG_ATTACKER_CKPT=./outputs/ascadv1_variable/strong_attacker/seed_0/best_val_rank.ckpt
    ;;
  1)
    DATASET=ascadv1_variable
    PATH_TO_EVAL=./outputs/ascadv1_variable/baselines/random.npy
    STRONG_ATTACKER_CKPT=./outputs/ascadv1_variable/strong_attacker/seed_0/best_val_rank.ckpt
    ;;
  2)
    DATASET=ches_ctf_2018
    PATH_TO_EVAL=./outputs/ches_ctf_2018/baselines/random.npy
    STRONG_ATTACKER_CKPT=./outputs/ches_ctf_2018/strong_attacker/seed_0/best_val_rank.ckpt
    ;;
esac

# ASCADv1-variable jobs: copy non-h5 dataset files to RAM for faster I/O
if [ "$DATASET" = "ascadv1_variable" ]; then
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

    EXTRA_ARGS="--ascadv1-variable-root $RAMFS_ROOT"
else
    EXTRA_ARGS=""
fi

source ~/.bashrc
micromamba activate leakage-localization
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python experiments/evaluate_trained_model.py \
    --path-to-eval "$PATH_TO_EVAL" \
    --dataset "$DATASET" \
    --strong-attacker-ckpt-path "$STRONG_ATTACKER_CKPT" \
    --metrics fwd-dnno-occl rev-dnno-occl \
    $EXTRA_ARGS