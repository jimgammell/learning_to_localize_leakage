import argparse
from pathlib import Path
from typing import Optional, get_args

from leakage_localization.datasets import DATASET, PARTITION

from init_things import *

def output_path(dataset_id: DATASET) -> Path:
    return OUTPUTS_ROOT / dash_to_uscr(dataset_id)

def fmt_dataset_name(dataset_id: DATASET) -> str:
    if dataset_id == 'ascadv1-fixed':
        return 'ASCADv1 (fixed key)'
    elif dataset_id == 'ascadv1-variable':
        return 'ASCADv1 (variable key)'
    elif dataset_id == 'ches-ctf-2018':
        return 'CHES-CTF-2018'
    else:
        assert False

def run_plot_training_curves(dest: Path):
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, WIDTH/3))
    for ax, dataset in zip(axes, get_args(DATASET)):
        attacker_path = None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--plot-training-curves', default=False, action='store_true'
    )
    parser.add_argument(
        '--plot-mtd-curves', default=False, action='store_true'
    )
    parser.add_argument(
        '--format-attack-performance', default=False, action='store_true'
    )
    parser.add_argument(
        '--dest', default=None, type=Path
    )
    args = parser.parse_args()

    plot_training_curves: bool = args.plot_training_curves
    assert isinstance(plot_training_curves, bool)
    plot_mtd_curves: bool = args.plot_mtd_curves
    assert isinstance(plot_mtd_curves, bool)
    format_attack_performance: bool = args.format_attack_performance
    assert isinstance(format_attack_performance, bool)
    dest: Optional[Path] = args.dest
    if dest is None:
        dest = OUTPUTS_ROOT / 'plots_for_paper'
    assert isinstance(dest, Path)
    dest.mkdir(exist_ok=True, parents=True)

    if plot_training_curves:
        run_plot_training_curves(dest / 'training_curves.pdf')

if __name__ == '__main__':
    main()