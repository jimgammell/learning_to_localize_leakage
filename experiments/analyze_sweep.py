from typing import Optional, List, Union
from pathlib import Path
from collections import defaultdict
import argparse

import pandas
import numpy as np
from matplotlib import pyplot as plt

from init_things import *

def compare_attack_localization_performance(sweep: pandas.DataFrame, dest: Path):
    attack_performance = 100*sweep['full_acc']
    localization_performance = sweep['full_oa']
    fig, axes = plt.subplots(1, 2, figsize=(2*WIDTH, WIDTH))
    axes[0].set_xlabel('Attack performance (full-key accuracy)')
    axes[0].set_ylabel('Localization performance (full-key white-box agreement)')
    axes[0].plot(attack_performance, localization_performance, color='blue', marker='.', linestyle='none', rasterized=True)
    axes[1].set_xlabel('Attack performance (byte 2 accuracy)')
    axes[1].set_ylabel('Localization performance (byte 2 white-box agreement)')
    attack_performance = 100*sweep['byte_acc']
    localization_performance = sweep['byte_oa']
    axes[1].plot(attack_performance, localization_performance, color='blue', marker='.', linestyle='none', rasterized=True)
    fig.tight_layout()
    fig.savefig(dest / 'attack_vs_loc_performance.pdf', dpi=DPI)
    plt.close(fig)

def plot_best_training_curves(sweep: pandas.DataFrame, dest: Path):
    best_attack_path, best_loc_path = get_best_model_paths(sweep)
    ametrics = pandas.read_csv(best_attack_path / 'metrics.csv')
    a_step = ametrics['step']
    a_train_loss = ametrics['train/loss']
    a_val_loss = ametrics['val/loss']
    a_train_acc = ametrics['train/acc']
    a_val_acc = ametrics['val/acc']
    lmetrics = pandas.read_csv(best_loc_path / 'metrics.csv')
    l_step = lmetrics['step']
    l_train_loss = lmetrics['train/loss']
    l_val_loss = lmetrics['val/loss']
    l_train_acc = lmetrics['train/acc']
    l_val_acc = lmetrics['val/acc']
    fig, axes = plt.subplots(1, 2, figsize=(2*WIDTH, WIDTH))
    axes[0].plot(a_step[~a_train_loss.isna()], a_train_loss[~a_train_loss.isna()], color='blue', linestyle=':', label='Best attacker (train)')
    axes[0].plot(a_step[~a_val_loss.isna()], a_val_loss[~a_val_loss.isna()], color='blue', linestyle='-', label='Best attacker (val)')
    axes[0].plot(l_step[~l_train_loss.isna()], l_train_loss[~l_train_loss.isna()], color='red', linestyle=':', label='Best localizer (train)')
    axes[0].plot(l_step[~l_val_loss.isna()], l_val_loss[~l_val_loss.isna()], color='red', linestyle='-', label='Best localizer (val)')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Cross-entropy loss (full key)')
    axes[0].set_yscale('log')
    axes[0].legend(loc='lower left', framealpha=0)
    axes[1].plot(a_step[~a_train_acc.isna()], a_train_acc[~a_train_acc.isna()], color='blue', linestyle=':', label='Best attacker (train)')
    axes[1].plot(a_step[~a_val_acc.isna()], a_val_acc[~a_val_acc.isna()], color='blue', linestyle='-', label='Best attacker (val)')
    axes[1].plot(l_step[~l_train_acc.isna()], l_train_acc[~l_train_acc.isna()], color='red', linestyle=':', label='Best localizer (train)')
    axes[1].plot(l_step[~l_val_acc.isna()], l_val_acc[~l_val_acc.isna()], color='red', linestyle='-', label='Best localizer (val)')
    axes[1].legend(loc='upper left', framealpha=0)
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Accuracy (full key)')
    fig.tight_layout()
    fig.savefig(dest / 'training_curve_comparison.pdf', dpi=DPI)
    plt.close(fig)

def get_best_model_paths(sweep: pandas.DataFrame) -> Union[Path, Path]:
    best_attack_idx = sweep['val_acc'].idxmax()
    best_attack_path = sweep['path'].loc[best_attack_idx]
    best_loc_idx = sweep['full_oa'].idxmax()
    best_loc_path = sweep['path'].loc[best_loc_idx]
    return best_attack_path, best_loc_path

def load_sweep(src: Path) -> pandas.DataFrame:
    run_paths: List[Path] = []
    for reg_dir in src.iterdir():
        if reg_dir.name == 'plots':
            continue
        if not reg_dir.is_dir():
            continue
        for seed_dir in reg_dir.iterdir():
            oracle_agreement_path = seed_dir / 'oracle_agreement.gradvis.npy'
            metrics_path = seed_dir / 'test_attack_metrics.npz'
            if not oracle_agreement_path.exists() or not metrics_path.exists():
                print(f'Skipping directory because of incomplete run: {seed_dir}')
            else:
                run_paths.append(seed_dir)
    results = defaultdict(
        lambda: np.full((len(run_paths),), np.nan, dtype=float)
    )
    results['path'] = [None for _ in range(len(run_paths))]
    for run_idx, run_path in enumerate(run_paths):
        results['path'][run_idx] = run_path
        attack_metrics = np.load(run_path / 'test_attack_metrics.npz')
        results['full_acc'][run_idx] = attack_metrics.get('test/acc', np.nan)
        results['byte_acc'][run_idx] = attack_metrics.get('test/acc/2', np.nan)
        results['full_mtd'][run_idx] = attack_metrics.get('test/mtd', np.nan)
        results['byte_mtd'][run_idx] = attack_metrics.get('test/mtd/2', np.nan)
        oracle_agreement = np.load(run_path / 'oracle_agreement.gradvis.npy')
        results['full_oa'][run_idx] = oracle_agreement.mean()
        results['byte_oa'][run_idx] = oracle_agreement[2]
        val_metrics = np.load(run_path / 'val_attack_metrics.npz')
        results['val_acc'][run_idx] = val_metrics.get('test/acc', np.nan)
    results = pandas.DataFrame(data=results)
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=Path, required=True)
    parser.add_argument('--dest', type=Path, default=None)
    args = parser.parse_args()

    src: Path = args.src
    assert src.exists()
    dest: Optional[Path] = args.dest
    if dest is None:
        dest = src / 'plots'
        dest.mkdir(exist_ok=True)
    
    sweep = load_sweep(src)
    print(sweep)
    compare_attack_localization_performance(sweep, dest)
    plot_best_training_curves(sweep, dest)

if __name__ == '__main__':
    main()