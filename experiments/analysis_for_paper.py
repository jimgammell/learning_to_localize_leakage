import argparse
from pathlib import Path
from typing import Optional, Literal, get_args
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from leakage_localization.datasets import DATASET, PARTITION
from leakage_localization.training.parse_metrics import parse_metrics

from init_things import *

def format_k(x: np.number, pos: Any) -> str:
    if x >= 1000:
        return f'{x/1000:.1f}'.rstrip('0').rstrip('.') + 'k'
    else:
        return f'{x:.0f}'

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

def fmt_metric_name(metric_id: Literal['acc', 'rank']) -> str:
    if metric_id == 'acc':
        return r'Accuracy (full key) $\uparrow$'
    elif metric_id == 'rank':
        return r'Rank (full key) $\downarrow$'
    else:
        assert False

def run_plot_training_curves(dest: Path):
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, WIDTH/3))
    for ax, dataset_id, metric_id in zip(axes, ['ascadv1-fixed', 'ascadv1-variable', 'ches-ctf-2018'], ['acc', 'acc', 'rank']):
        attack_runs_path = output_path(dataset_id) / 'strong_attacker'
        train_metrics, val_metrics = defaultdict(list), defaultdict(list)
        for seed in [0, 1, 2, 3, 4]:
            _train_metrics, _val_metrics = parse_metrics(attack_runs_path / f'seed_{seed}' / 'metrics.csv')
            for k, v in _train_metrics.items():
                train_metrics[k].append(v)
            for k, v in _val_metrics.items():
                val_metrics[k].append(v)
        train_metrics = {k: np.stack(v) for k, v in train_metrics.items()}
        val_metrics = {k: np.stack(v) for k, v in val_metrics.items()}
        train_steps = train_metrics['step']
        val_steps = val_metrics['step']
        train_steps = train_steps[0, :]
        val_steps = val_steps[0, :]
        ax.set_xlabel('Training step')
        ax.set_ylabel(f'{fmt_metric_name(metric_id)}')
        ax.set_title(f'{fmt_dataset_name(dataset_id)}')
        ax.fill_between(
            train_steps, train_metrics[metric_id].min(axis=0), train_metrics[metric_id].max(axis=0),
            color='grey', alpha=0.25, rasterized=True
        )
        ax.plot(train_steps, np.median(train_metrics[metric_id], axis=0), color='grey', label='train', rasterized=True)
        ax.fill_between(
            val_steps, val_metrics[metric_id].min(axis=0), val_metrics[metric_id].max(axis=0),
            color='blue', alpha=0.25, rasterized=True
        )
        ax.plot(val_steps, np.median(val_metrics[metric_id], axis=0), color='blue', label='val', rasterized=True)
        ax.legend(framealpha=0)
        ax.xaxis.set_major_formatter(FuncFormatter(format_k))
        if dataset_id == 'ascadv1-fixed':
            ax.xaxis.set_major_locator(MultipleLocator(10_000))
        elif dataset_id == 'ascadv1-variable':
            ax.xaxis.set_major_locator(MultipleLocator(25_000))
        elif dataset_id == 'ches-ctf-2018':
            ax.xaxis.set_major_locator(MultipleLocator(2_500))
        fig.tight_layout()
        fig.savefig(dest, dpi=DPI)
        plt.close(fig)

def run_plot_mtd_curves(dest: Path):
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, WIDTH/3))
    for ax, dataset_id in zip(axes, ['ascadv1-fixed', 'ascadv1-variable', 'ches-ctf-2018']):
        attack_runs_path = output_path(dataset_id) / 'strong_attacker'
        mtd_curves = np.full((5, 16, 1000), np.nan, dtype=float)
        for seed in [0, 1, 2, 3, 4]:
            attack_metrics_path = attack_runs_path / f'seed_{seed}' / 'attack_metrics.npz'
            attack_metrics = np.load(attack_metrics_path, allow_pickle=True)
            mtd_curve = attack_metrics['rank_over_time']
            mtd_curves[seed, :, :] = mtd_curve
        ax.set_xlabel('Traces seen')
        ax.set_ylabel(r'Rank (per-byte) $\downarrow$')
        ax.set_title(f'{fmt_dataset_name(dataset_id)}')
        mtd_curves = mtd_curves.reshape(-1, 1000)
        traces_seen = np.arange(1, 1001)
        ax.fill_between(
            traces_seen, mtd_curves.min(axis=0), mtd_curves.max(axis=0),
            color='blue', alpha=0.25, rasterized=True
        )
        ax.plot(traces_seen, np.median(mtd_curves, axis=0), color='blue', rasterized=True)
        ax.set_xscale('log')
        fig.tight_layout()
        fig.savefig(dest, dpi=DPI)
        plt.close(fig)

def run_plot_cost_scaling(dest: Path):
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH, WIDTH/4))
    benchmark_path = OUTPUTS_ROOT / 'compute_benchmark' / 'results.npz'
    benchmark = np.load(benchmark_path, allow_pickle=True)

    sweep_var     = benchmark['sweep_var']
    param_count   = benchmark['param_count']
    flops         = benchmark['flops']
    wall_time_ms  = benchmark['wall_time_ms']    # (n_configs, N_SEEDS)
    vram_gb       = benchmark['vram_mb'] / 1024  # (n_configs, N_SEEDS)

    # Each sweep's x-values are the raw parameter values for that sweep's rows,
    # normalised to the middle entry (the base configuration).
    sweep_cfgs = {
        'embedding_dim': ('Hidden dim (base=256)',    benchmark['embedding_dim']),
        'layer_count':   ('Layer count (base=4)',   benchmark['layer_count']),
        'patch_count':   ('Patch count (base=32)',   benchmark['patch_count']),
    }
    colors  = ['red', 'blue', 'green']

    # (metric_data, ylabel, has_seeds) — seeded metrics get a min/max band
    panel_specs = [
        (param_count,  r'Parameters',           False),
        (flops,        r'FLOPs/step',            False),
        (vram_gb,      r'VRAM [GB]',            True),
        (wall_time_ms, r'Time/step [A6000-ms]', True),
    ]

    for ax, (metric_data, ylabel, has_seeds) in zip(axes, panel_specs):
        for color, (sv_key, (sv_label, sv_raw)) in zip(colors, sweep_cfgs.items()):
            mask = sweep_var == sv_key
            if not mask.any():
                continue
            x = sv_raw[mask].astype(float)
            x = x / x[len(x) // 2]   # normalise: base → 1, neighbours → 0.5/2, …

            if has_seeds:
                y = metric_data[mask]                      # (n_pts, N_SEEDS)
                ax.plot(x, np.mean(y, axis=1), color=color, marker='.',
                        linewidth=0.5, markersize=3, label=sv_label, rasterized=True)
            else:
                ax.plot(x, metric_data[mask], color=color, marker='.',
                        linewidth=0.5, markersize=3, label=sv_label, rasterized=True)

        ax.set_xlabel('Hyperparameter/base')
        ax.set_ylabel(ylabel)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(sweep_cfgs),
               framealpha=0, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    fig.savefig(dest, dpi=DPI, bbox_inches='tight')
    plt.close(fig)


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
        '--plot-cost-scaling', default=False, action='store_true'
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
    plot_cost_scaling: bool = args.plot_cost_scaling
    assert isinstance(plot_cost_scaling, bool)
    dest: Optional[Path] = args.dest
    if dest is None:
        dest = OUTPUTS_ROOT / 'plots_for_paper'
    assert isinstance(dest, Path)
    dest.mkdir(exist_ok=True, parents=True)

    if plot_training_curves:
        run_plot_training_curves(dest / 'training_curves.pdf')
    if plot_mtd_curves:
        run_plot_mtd_curves(dest / 'mtd_curves.pdf')
    if plot_cost_scaling:
        run_plot_cost_scaling(dest / 'cost_scaling.pdf')

if __name__ == '__main__':
    main()