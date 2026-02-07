#!/bin/bash

#SBATCH --job-name=train-ascadv1f-perceiver
#SBATCH --partition=cocosys
#SBATCH --account=cocosys
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=14
#SBATCH --time=24:00:00
#SBATCH --output=train-ascadv1f-perceiver-%a.out
#SBATCH --error=train-ascadv1f-perceiver-%a.out
#SBATCH --array=0-7

#                         roll  noise  droppatch  dropout
ROLL=(                    0     100    0          0        0      0      100    100)
NOISE=(                   0.    0.     0.1        0.       0.     0.     0.1    0.1)
INPUT_DROPOUT=(           0.    0.     0.         0.2      0.     0.     0.2    0.2)
HIDDEN_DROPOUT=(          0.    0.     0.         0.       0.1    0.     0.     0.1)
NAMES=(baseline roll noise droppatch dropout roll+noise roll+noise+droppatch all)

IDX=$SLURM_ARRAY_TASK_ID

source ~/.bashrc
micromamba activate leakage-localization
python -m experiments.train.supervised \
    --dest=./outputs/ascadv1_fixed/supervised_reg_${NAMES[$IDX]} \
    --config-file=ascadv1_fixed \
    --training.base_lr 3.e-4 \
    --data.random_roll ${ROLL[$IDX]} \
    --data.additive_gaussian_noise ${NOISE[$IDX]} \
    --model.input_dropout_rate ${INPUT_DROPOUT[$IDX]} \
    --model.hidden_dropout_rate ${HIDDEN_DROPOUT[$IDX]}