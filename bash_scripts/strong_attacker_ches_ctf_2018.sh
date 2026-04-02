#!/bin/bash

for seed in 0 1 2 3 4; do
    bash ./bash_scripts/sup_train_and_eval.sh \
        ches_ctf_2018 \
        ./outputs/ches_ctf_2018/strong_attacker/seed_${seed} \
        ./outputs/ches_ctf_2018/strong_attacker/seed_0/best_val_rank.ckpt \
        --training.seed ${seed}
done
