#!/bin/bash
# This script streams a subset of our results from the source directory to a destination directory.
# We are selectively streaming results according to the following criteria:
#   1) Users should be able to selectively run individual experiments without having to run e.g. hyperparameter tuning first.
#   2) We want users to be able to re-generate the important plots from our paper without training models.

src=$1
dest=$2

rsync -av \
    --exclude "*/lightning_output" \
    --exclude "*.png" \
    --exclude "*.pdf" \
    --exclude "*ablation" \
    --exclude "*all_sensitivity_analysis" \
    --exclude "*backup*" \
    $src $dest