for seed in 0 1 2 3 4; do
    python -m experiments.train.supervised \
        --dest ./outputs/ascadv1_fixed/multiseed/seed_${seed} \
        --config-file ascadv1_fixed \
        --training.seed ${seed}
done