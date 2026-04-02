import argparse
from pathlib import Path
from typing import List
from collections import defaultdict
from math import log

import pandas
import numpy as np
from matplotlib import pyplot as plt

from leakage_localization.evaluation.mtd import compute_mtd

from init_things import *
from utils.visualize_runs import *

def load_sweep(sweep_dir: Path) -> pandas.DataFrame:
    trial_dirs: List[Path] = []
    for x in sweep_dir.iterdir():
        if not x.is_dir():
            continue
        if not 'trial_' in x.name:
            continue
        if not (x / 'metrics.csv').exists():
            continue
        trial_dirs.append(x)
    trial_dirs.sort(key=lambda x: x.name)

    data = defaultdict(list)
    for trial_dir in trial_dirs:
        data['path'].append(trial_dir)
        attack_metrics_path = trial_dir / 'attack_metrics.npz'
        assert attack_metrics_path.exists(), attack_metrics_path
        attack_metrics = np.load(attack_metrics_path, allow_pickle=True)
        for metric in ['loss', 'rank', 'acc']:
            data[metric].append(attack_metrics[f'test/{metric}'].item())
            for byte_idx in range(16):
                data[f'{metric}/{byte_idx}'].append(attack_metrics[f'test/{metric}/{byte_idx}'].item())
        data['mtd'].append(attack_metrics['test/mtd'].item())
        for byte_idx in range(16):
            data[f'mtd/{byte_idx}'].append(attack_metrics['per_byte_mtd'][byte_idx])
        for attr_method in ['gradvis', 'input_x_gradient']:
            # fwd DNN occlusion — new format is .npz, old format is .npy
            fwd_dnno_npz = trial_dir / f'fwd_dnno_occl.{attr_method}.npz'
            fwd_dnno_npy = trial_dir / f'fwd_dnno_occl.{attr_method}.npy'
            assert fwd_dnno_npz.exists() or fwd_dnno_npy.exists(), fwd_dnno_npz
            if fwd_dnno_npz.exists():
                fwd_dnno_data = np.load(fwd_dnno_npz, allow_pickle=True)
                data[f'fwd_dnno/{attr_method}'].append(fwd_dnno_data['fwd-dnno-occl'].mean())
                b2_key = 'fwd-dnno-occl/2'
                data[f'fwd_dnno/{attr_method}/2'].append(fwd_dnno_data[b2_key].mean() if b2_key in fwd_dnno_data else np.nan)
            else:
                fwd_dnno = np.load(fwd_dnno_npy)
                data[f'fwd_dnno/{attr_method}'].append(fwd_dnno.mean())
                data[f'fwd_dnno/{attr_method}/2'].append(np.nan)
            # rev DNN occlusion
            rev_dnno_npz = trial_dir / f'rev_dnno_occl.{attr_method}.npz'
            rev_dnno_npy = trial_dir / f'rev_dnno_occl.{attr_method}.npy'
            assert rev_dnno_npz.exists() or rev_dnno_npy.exists(), rev_dnno_npz
            if rev_dnno_npz.exists():
                rev_dnno_data = np.load(rev_dnno_npz, allow_pickle=True)
                data[f'rev_dnno/{attr_method}'].append(rev_dnno_data['rev-dnno-occl'].mean())
                b2_key = 'rev-dnno-occl/2'
                data[f'rev_dnno/{attr_method}/2'].append(rev_dnno_data[b2_key].mean() if b2_key in rev_dnno_data else np.nan)
            else:
                rev_dnno = np.load(rev_dnno_npy)
                data[f'rev_dnno/{attr_method}'].append(rev_dnno.mean())
                data[f'rev_dnno/{attr_method}/2'].append(np.nan)
            # TA MTD — new format has 'ta-mtd' (full-key) and 'ta-mtd/{b}' (per-byte)
            ta_mtd_path = trial_dir / f'ta_mtd.{attr_method}.npz'
            assert ta_mtd_path.exists(), ta_mtd_path
            ta_mtd_data = np.load(ta_mtd_path, allow_pickle=True)
            if 'ta-mtd' in ta_mtd_data:
                data[f'ta_mtd/{attr_method}'].append(float(ta_mtd_data['ta-mtd']))
                for byte_idx in range(16):
                    data[f'ta_mtd/{attr_method}/{byte_idx}'].append(float(ta_mtd_data[f'ta-mtd/{byte_idx}']))
            else:
                # old format: 'mtd' is the per-byte array; no full-key MTD saved
                data[f'ta_mtd/{attr_method}'].append(np.nan)
                for byte_idx in range(16):
                    data[f'ta_mtd/{attr_method}/{byte_idx}'].append(ta_mtd_data['mtd'][byte_idx])
            # white-box agreement
            white_box_path = trial_dir / f'white_box_agreement.{attr_method}.npz'
            assert white_box_path.exists(), white_box_path
            white_box = np.load(white_box_path, allow_pickle=True)
            for byte_idx in range(16):
                data[f'white_box_spearman/{attr_method}/{byte_idx}'].append(white_box['spearman'][byte_idx])
                data[f'white_box_auroc/{attr_method}/{byte_idx}'].append(white_box['auroc'][byte_idx])
            data[f'white_box_spearman/{attr_method}/full'].append(float(white_box['full_spearman']) if 'full_spearman' in white_box else np.nan)
            data[f'white_box_auroc/{attr_method}/full'].append(float(white_box['full_auroc']) if 'full_auroc' in white_box else np.nan)
    data = pandas.DataFrame(data)
    for attr_method in ['gradvis', 'input_x_gradient']:
        data[f'white_box_spearman/{attr_method}'] = data[[f'white_box_spearman/{attr_method}/{byte_idx}' for byte_idx in range(16)]].mean(axis=1)
        data[f'white_box_auroc/{attr_method}'] = data[[f'white_box_auroc/{attr_method}/{byte_idx}' for byte_idx in range(16)]].mean(axis=1)
    return data

def get_best_attacker(sweep: pandas.DataFrame) -> Path:
    best_row = sweep.loc[sweep['acc'].idxmax()]
    best_path = best_row['path']
    return best_path

def get_best_localizer(sweep: pandas.DataFrame) -> Path:
    best_row = sweep.loc[sweep['white_box_spearman/gradvis'].idxmax()]
    best_path = best_row['path']
    return best_path

# Rows: metrics + methods
# Columns: datasets
def tabulate_best_performance(sweep: pandas.DataFrame, dest: Path):
    rv: List[str] = []
    rv.append(r'\begin{tabular}{c lccc}')
    rv.append(r'\toprule')
    rv.append(r'& ASCADv1 (fixed) & ASCADv1 (variable) & CHES-CTF-2018 \\')

def run_plot_training_curves(sweep: pandas.DataFrame, dest: Path):
    best_attacker_path = get_best_attacker(sweep)
    best_localizer_path = get_best_localizer(sweep)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, WIDTH/2))
    plot_training_curves(
        best_attacker_path, axes[0], 'loss', color='red',
        train_plot_kwargs={'label': 'Best attacker (train)'},
        val_plot_kwargs={'label': 'Best attacker (val)'}
    )
    plot_training_curves(
        best_localizer_path, axes[0], 'loss', color='blue',
        train_plot_kwargs={'label': 'Best localizer (train)'},
        val_plot_kwargs={'label': 'Best localizer (val)'}
    )
    plot_training_curves(
        best_attacker_path, axes[1], 'rank', color='red',
        train_plot_kwargs={'label': 'Best attacker (train)'},
        val_plot_kwargs={'label': 'Best attacker (val)'}
    )
    plot_training_curves(
        best_localizer_path, axes[1], 'rank', color='blue',
        train_plot_kwargs={'label': 'Best localizer (train)'},
        val_plot_kwargs={'label': 'Best localizer (val)'}
    )
    random_loss = log(256)
    axes[0].set_ylim(0, 1.1*random_loss)
    random_rank = 0.5*(1 + 256)
    axes[1].set_ylim(0, 1.1*random_rank)
    axes[0].set_xlabel('Training step')
    axes[0].set_ylabel(r'Cross-entropy loss $\downarrow$')
    axes[1].set_xlabel('Training step')
    axes[1].set_ylabel(r'Rank $\downarrow$')
    axes[0].legend(loc='upper right', framealpha=0., fontsize=6)
    axes[1].legend(loc='upper right', framealpha=0., fontsize=6)
    fig.tight_layout()
    fig.savefig(dest, dpi=DPI)
    plt.close(fig)

def run_plot_mtd(sweep: pandas.DataFrame, dest: Path):
    best_attacker_path = get_best_attacker(sweep)
    best_localizer_path = get_best_localizer(sweep)
    fig, ax = plt.subplots(1, 1, figsize=(WIDTH/2, WIDTH/2))
    plot_mtd(best_attacker_path, ax, color='red', worst_byte_kwargs=dict(label='Best attacker'))
    plot_mtd(best_localizer_path, ax, color='blue', worst_byte_kwargs=dict(label='Best localizer'))
    ax.set_xlabel('Traces seen')
    ax.set_ylabel('Byte rank')
    ax.legend(loc='upper right', framealpha=0, fontsize=6)
    ax.set_xscale('log')
    fig.tight_layout()
    fig.savefig(dest, dpi=DPI)
    plt.close(fig)

def run_white_box_agreement(sweep: pandas.DataFrame, dest: Path):
    best_attacker_path = get_best_attacker(sweep)
    best_localizer_path = get_best_localizer(sweep)

def run_plot_gradvis_vs_inputxgrad(sweep: pandas.DataFrame, dest: Path):
    pass

def run_plot_attack_vs_loc(sweep: pandas.DataFrame, dest: Path):
    with plt.rc_context({'font.size': 6, 'axes.labelsize': 6, 'xtick.labelsize': 5, 'ytick.labelsize': 5}):
        fig, axes = plt.subplots(1, 4, figsize=(WIDTH, WIDTH/4))
        kwargs = dict(
            color='blue',
            marker='.',
            linestyle='none',
            markersize=5
        )
        axes[0].plot(sweep['acc/2'], sweep['white_box_auroc/gradvis/2'], **kwargs)
        axes[1].plot(sweep['acc/2'], sweep['fwd_dnno/gradvis'], **kwargs)
        axes[2].plot(sweep['acc/2'], sweep['rev_dnno/gradvis'], **kwargs)
        axes[3].plot(sweep['acc/2'], sweep['ta_mtd/gradvis/2'], **kwargs)
        axes[3].set_yscale('log')
        axes[0].set_xlabel('Accuracy')
        axes[1].set_xlabel('Accuracy')
        axes[2].set_xlabel('Accuracy')
        axes[3].set_xlabel('Accuracy')
        axes[0].set_ylabel('White box AUROC')
        axes[1].set_ylabel('Forward DNN occlusion')
        axes[2].set_ylabel('Reverse DNN occlusion')
        axes[3].set_ylabel('Tempalate attack MTD')
        for ax in axes:
            ax.tick_params(axis='both', which='both', pad=2)
        fig.tight_layout()
        fig.savefig(dest, dpi=DPI)
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--sweep-dir', type=Path, required=True,
        help='Base directory of the sweep to be plotted.'
    )
    parser.add_argument(
        '--dest', type=Path, default=None,
        help='Directory in which to save figures. Defaults to a directory called `plots` in the sweep directory.'
    )
    args = parser.parse_args()

    sweep_dir: Path = args.sweep_dir
    assert isinstance(sweep_dir, Path) and sweep_dir.exists()
    dest: Optional[Path] = args.dest
    if dest is None:
        dest = sweep_dir / 'plots'
        dest.mkdir(exist_ok=True)
    assert isinstance(dest, Path) and dest.exists()

    sweep = load_sweep(sweep_dir)
    print(f'Best attacker path: {get_best_attacker(sweep)}')
    print(f'Best localizer path: {get_best_localizer(sweep)}')
    
    # table listing performance of the best attacker and localizer models

    # training curves for the best attacker and best localizer
    run_plot_training_curves(sweep, dest / 'training_curves.pdf')

    # rank over time for the best attacker and best localizer
    run_plot_mtd(sweep, dest / 'mtd.pdf')

    # leakiness over time visualizations for oracle, best attacker, best localizer

    # visualizations of the DNN occlusion tests for the oracle, random, best attacker, best localizer

    # visualizations of template attack MTD for the oracle, random, best attacker, best localizer

    # scatterplots showing relationship between the different attack/localization performance metrics
    run_plot_attack_vs_loc(sweep, dest / 'attack_vs_loc.pdf')

    # scatterplots showing relationship between GradVis and input x grad

if __name__ == '__main__':
    main()