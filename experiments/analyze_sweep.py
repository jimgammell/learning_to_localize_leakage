from typing import Optional, List, Union
from pathlib import Path
from collections import defaultdict
import argparse

import pandas
import numpy as np
from matplotlib import pyplot as plt

from init_things import *
from utils.visualize_runs import (
    plot_training_curves,
    plot_occlusion_test,
    plot_template_attack_test,
    plot_white_box_agreement
)

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

def run_plot_best_training_curves(sweep: pandas.DataFrame, dest: Path):
    best_attack_path, best_loc_path = get_best_model_paths(sweep)
    fig, axes = plt.subplots(1, 2, figsize=(2*WIDTH, WIDTH))
    plot_training_curves(
        best_attack_path,
        axes[0],
        'loss',
        color='red',
        train_plot_kwargs={'label': 'Best attacker (train)'},
        val_plot_kwargs={'label': 'Best attacker (val)'}
    )
    plot_training_curves(
        best_loc_path,
        axes[0],
        'loss',
        color='blue',
        train_plot_kwargs={'label': 'Best localizer (train)'},
        val_plot_kwargs={'label': 'Best localizer (val)'}
    )
    plot_training_curves(
        best_attack_path,
        axes[1],
        'acc',
        color='red',
        train_plot_kwargs={'label': 'Best attacker (train)'},
        val_plot_kwargs={'label': 'Best attacker (val)'}
    )
    plot_training_curves(
        best_loc_path,
        axes[1],
        'acc',
        color='blue',
        train_plot_kwargs={'label': 'Best localizer (train)'},
        val_plot_kwargs={'label': 'Best localizer (val)'}
    )
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Cross-entropy loss (full key)')
    axes[0].set_yscale('log')
    axes[0].legend(loc='lower left', framealpha=0)
    axes[1].legend(loc='upper left', framealpha=0)
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Accuracy (full key)')
    fig.tight_layout()
    fig.savefig(dest / 'training_curve_comparison.pdf', dpi=DPI)
    plt.close(fig)

def run_plot_occlusion_test(src: Path, dest: Path):
    fwd_path = src / 'fwd_dnno_occl.gradvis.npy'
    rev_path = src / 'rev_dnno_occl.gradvis.npy'
    fig, ax = plt.subplots(1, 1, figsize=(WIDTH, WIDTH))
    plot_occlusion_test(
        fwd_path,
        ax,
        color='blue',
        label='Forward'
    )
    plot_occlusion_test(
        rev_path,
        ax,
        color='red',
        label='Reverse'
    )
    ax.set_xlabel('Points occluded')
    ax.set_ylabel('MTD (full key)')
    ax.set_title('Visualization of DNN occlusion tests')
    ax.legend(framealpha=0, loc='upper right')
    fig.tight_layout()
    fig.savefig(dest / 'occlusion_test_vis.pdf', dpi=DPI)
    plt.close(fig)

def run_plot_template_attack_test(src: Path, dest: Path):
    ta_path = src / 'ta_mtd.gradvis.npz'
    fig, ax = plt.subplots(1, 1, figsize=(WIDTH, WIDTH))
    plot_template_attack_test(
        ta_path,
        ax,
    )
    ax.set_xlabel('Traces seen')
    ax.set_ylabel('MTD (full key)')
    ax.set_title('Visualization of template attack performance')
    ax.set_xscale('log')
    fig.tight_layout()
    fig.savefig(dest / 'template_attack_test.pdf', dpi=DPI)
    plt.close(fig)

def run_plot_white_box_agreement(black_box_src: Path, white_box_src: Path, dest: Path):
    fig = plt.figure(figsize=(WIDTH, WIDTH/4))
    gs = fig.add_gridspec(2, 9, width_ratios=[1, 1, 0.2, 1, 1, 0.2, 1, 1, 1], height_ratios=[1, 1], wspace=0.25, hspace=0.25)
    vars_ax = fig.add_subplot(gs[0:2, 6:9])
    vars_tax = vars_ax.twinx()
    oracle_ax = fig.add_subplot(gs[0:2, 0:2])
    measured_ax = fig.add_subplot(gs[0:2, 3:5])
    rin_ax = fig.add_subplot(gs[0, 6])
    srin_ax = fig.add_subplot(gs[0, 7], sharex=rin_ax)
    r_ax = fig.add_subplot(gs[1, 6], sharex=rin_ax)
    sr_ax = fig.add_subplot(gs[1, 7], sharex=rin_ax)
    rout_ax = fig.add_subplot(gs[0, 8], sharex=rin_ax)
    srout_ax = fig.add_subplot(gs[1, 8], sharex=rin_ax)
    var_axes = np.array([rin_ax, srin_ax, r_ax, sr_ax, rout_ax, srout_ax])

    plot_white_box_agreement(
        black_box_src / 'gradvis.npy',
        white_box_src,
        oracle_ax,
        measured_ax,
        var_axes
    )
    oracle_ax.set_xlabel(r'Time $t$')
    oracle_ax.set_ylabel(r'White-box SNR of $X_t$')
    oracle_ax.set_yscale('log')
    fig.savefig(dest / 'white_box_comparison.pdf', dpi=DPI, bbox_inches='tight')
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
    run_plot_best_training_curves(sweep, dest)
    run_plot_occlusion_test(
        Path(r'/home/jgammell/leakage-localization-publishable/outputs/ascadv1_fixed/reg_sweep/gaussian_noise_0./seed_0'),
        dest
    )
    run_plot_template_attack_test(
        Path(r'/home/jgammell/leakage-localization-publishable/outputs/ascadv1_fixed/reg_sweep/gaussian_noise_0./seed_0'),
        dest
    )
    run_plot_white_box_agreement(
        Path(r'/home/jgammell/leakage-localization-publishable/outputs/ascadv1_fixed/reg_sweep/gaussian_noise_0./seed_0'),
        Path(r'/home/jgammell/leakage-localization-publishable/outputs/ascadv1_fixed/snr'),
        dest
    )

if __name__ == '__main__':
    main()