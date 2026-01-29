#!/bin/bash

for dataset in ascadv1-fixed ascadv1-variable
do
    for partition in profile attack
    do
        python -m experiments.snr.compute --dataset=$dataset --partition=$partition
        python -m experiments.snr.visualize --dataset=$dataset --partition=$partition
    done
done