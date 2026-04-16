import argparse
from pathlib import Path
from typing import Optional, Literal, List, Tuple, get_args
from collections import defaultdict
from tqdm import tqdm
import yaml
from math import ceil

import pandas
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from matplotlib.lines import Line2D
import re
from leakage_localization.datasets import DATASET, PARTITION
from leakage_localization.training.parse_metrics import parse_metrics
from leakage_localization.evaluation import OracleAgreement

from init_things import *
from utils.visualize_runs import *

FEATURE_COUNTS = {
    'ascadv1-fixed': 100_000,
    'ascadv1-variable': 250_000,
    'ches-ctf-2018': 650_000
}

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
        return r'Rank (avg. per-byte) $\downarrow$'
    else:
        assert False

def run_plot_cost_scaling(dest: Path):
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH, WIDTH/4))
    benchmark_path = OUTPUTS_ROOT / 'compute_benchmark' / 'results.npz'
    benchmark = np.load(benchmark_path, allow_pickle=True)

    sweep_var    = benchmark['sweep_var']
    param_count  = benchmark['param_count']
    flops        = benchmark['flops']
    wall_time_ms = benchmark['wall_time_ms']
    vram_gb      = benchmark['vram_mb'] / 1024

    # Base values used when a parameter is held fixed — x is normalised to these.
    base_vals = {'patch_count': 64, 'layer_count': 8, 'embedding_dim': 512}

    sweep_cfgs = {
        'embedding_dim': ('Hidden dim (base=512)',  benchmark['embedding_dim']),
        'layer_count':   ('Layer count (base=8)',   benchmark['layer_count']),
        'patch_count':   ('Patch count (base=64)',  benchmark['patch_count']),
    }
    colors = ['red', 'blue', 'green']

    panel_specs = [
        (param_count / 1e6,  r'Parameters (M)',   FuncFormatter(lambda v, _: f'{v:.0f}M')),
        (flops / 1e12,       r'TFLOPs/step',      FuncFormatter(lambda v, _: f'{v:.0f}T')),
        (vram_gb,            r'VRAM [GB]',         None),
        (wall_time_ms,       r'Time/step [ms]',   None),
    ]

    for ax, (metric_data, ylabel, yfmt) in zip(axes, panel_specs):
        for color, (sv_key, (sv_label, sv_raw)) in zip(colors, sweep_cfgs.items()):
            mask = sweep_var == sv_key
            if not mask.any():
                continue
            x = sv_raw[mask].astype(float) / base_vals[sv_key]
            ax.plot(x, metric_data[mask], color=color, marker='none',
                    linewidth=.75, label=sv_label, rasterized=True)

        if yfmt is not None:
            ax.yaxis.set_major_formatter(yfmt)
        ax.set_xlabel('Fraction of base')
        ax.set_ylabel(ylabel)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(sweep_cfgs),
               framealpha=0, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    fig.savefig(dest, dpi=DPI, bbox_inches='tight')
    plt.close(fig)

def load_sweep(sweep_dir: Path, dataset_id: DATASET) -> pandas.DataFrame:
    if not (sweep_dir / 'sweep_summary.csv').exists():
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
        for trial_dir in tqdm(trial_dirs):
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
                if ta_mtd_path.exists():
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
                else:
                    data[f'ta_mtd/{attr_method}'].append(float('nan'))
                    for byte_idx in range(16):
                        data[f'ta_mtd/{attr_method}/{byte_idx}'].append(float('nan'))
                # white-box agreement (not available for all datasets)
                snr_dir = get_output_dir(dataset_id) / 'snr'
                if snr_dir.exists():
                    oracle_agreement = OracleAgreement(snr_dir, dataset_id)
                    leakiness_estimates = np.load(trial_dir / f'{attr_method}.npy')
                    data[f'white_box_spearman/{attr_method}'].append(oracle_agreement.get_full_spearman(leakiness_estimates))
                    data[f'white_box_auroc/{attr_method}'].append(oracle_agreement.get_full_auroc(leakiness_estimates))
                    per_byte_spearman = oracle_agreement(leakiness_estimates)
                    per_byte_auroc = oracle_agreement.get_auroc(leakiness_estimates)
                    for byte_idx in range(16):
                        data[f'white_box_spearman/{attr_method}/{byte_idx}'].append(per_byte_spearman[byte_idx])
                        data[f'white_box_auroc/{attr_method}/{byte_idx}'].append(per_byte_auroc[byte_idx])
                else:
                    data[f'white_box_spearman/{attr_method}'].append(float('nan'))
                    data[f'white_box_auroc/{attr_method}'].append(float('nan'))
                    for byte_idx in range(16):
                        data[f'white_box_spearman/{attr_method}/{byte_idx}'].append(float('nan'))
                        data[f'white_box_auroc/{attr_method}/{byte_idx}'].append(float('nan'))
        data = pandas.DataFrame(data)
        data['mean_acc'] = data[[f'acc/{byte_idx}' for byte_idx in range(16)]].mean(axis=1)
        for attr_method in ['gradvis', 'input_x_gradient']:
            data[f'mean_ta_mtd/{attr_method}'] = data[[f'ta_mtd/{attr_method}/{byte_idx}' for byte_idx in range(16)]].mean(axis=1)
        data.to_csv(sweep_dir / 'sweep_summary.csv')
    data = pandas.read_csv(sweep_dir / 'sweep_summary.csv')
    return data

def get_best_loc_ches(sweep: pandas.DataFrame) -> pandas.Series:
    """Pick best localizer for CHES-CTF-2018 by average rank across 3 metrics.
    fwd_dnno and ta_mtd: lower is better. rev_dnno: higher is better."""
    fwd = sweep['fwd_dnno/input_x_gradient']
    rev = sweep['rev_dnno/input_x_gradient']
    ta  = sweep[[f'ta_mtd/input_x_gradient/{b}' for b in range(16)]].mean(axis=1)
    rank_fwd = fwd.rank(ascending=True)
    rank_rev = rev.rank(ascending=False)
    rank_ta  = ta.rank(ascending=True)
    avg_rank = (rank_fwd + rank_rev + rank_ta) / 3
    return sweep.loc[avg_rank.idxmin()]

def get_best_runs(dataset_id: DATASET) -> Tuple[pandas.Series, ...]:
    sweep_path = get_output_dir(dataset_id) / 'htune_highdropout'
    sweep = load_sweep(sweep_path, dataset_id)
    if dataset_id == 'ches-ctf-2018':
        mean_mtd = sweep[[f'mtd/{b}' for b in range(16)]].mean(axis=1)
        best_attack = sweep.loc[mean_mtd.idxmin()]
        best_loc = get_best_loc_ches(sweep)
    else:
        acc = sweep['mean_acc']
        best_attack = sweep.loc[acc.idxmax()]
        loc_metric = sweep[[f'white_box_auroc/input_x_gradient/{byte_idx}' for byte_idx in range(16)]].mean(axis=1)
        best_loc = sweep.loc[loc_metric.idxmax()]
    return best_attack, best_loc

def run_plot_training_curves(dest: Path):
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, WIDTH/3), layout='constrained')
    for ax, dataset_id, metric_id in zip(axes, ['ascadv1-fixed', 'ascadv1-variable', 'ches-ctf-2018'], ['acc', 'acc', 'rank']):
        best_attack_rv, best_loc_rv = get_best_runs(dataset_id)
        best_attack_path = Path(best_attack_rv['path'])
        best_loc_path = Path(best_loc_rv['path'])
        attack_train_metrics, attack_val_metrics = parse_metrics(best_attack_path / 'metrics.csv')
        ax.plot(attack_train_metrics['step'], attack_train_metrics[metric_id], color='red', linestyle=':', label='Best attacker (train)', rasterized=True)
        ax.plot(attack_val_metrics['step'], attack_val_metrics[metric_id], color='red', linestyle='-', label='Best attacker (val)', rasterized=True)
        loc_train_metrics, loc_val_metrics = parse_metrics(best_loc_path / 'metrics.csv')
        ax.plot(loc_train_metrics['step'], loc_train_metrics[metric_id], color='blue', linestyle=':', label='Best localizer (train)', rasterized=True)
        ax.plot(loc_val_metrics['step'], loc_val_metrics[metric_id], color='blue', linestyle='-', label='Best localizer (val)', rasterized=True)
        ax.set_xlabel('Training step')
        ax.set_ylabel(f'{fmt_metric_name(metric_id)}')
        ax.set_title(f'{fmt_dataset_name(dataset_id)}')
        ax.ticklabel_format(style='sci', axis='x', scilimits=(-2, 2), useMathText=True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncols=4, framealpha=0, bbox_to_anchor=(0.5, 0))
    fig.get_layout_engine().set(rect=(0, 0.15, 1, 1))
    fig.savefig(dest, dpi=DPI, bbox_inches='tight')
    plt.close(fig)

def run_plot_mtd_curves(dest: Path):
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, WIDTH/3), layout='constrained')
    lw = 0.75
    for ax, dataset_id in zip(axes, ['ascadv1-fixed', 'ascadv1-variable', 'ches-ctf-2018']):
        best_attack_rv, best_loc_rv = get_best_runs(dataset_id)
        best_attack_path = Path(best_attack_rv['path'])
        best_loc_path = Path(best_loc_rv['path'])
        attack_mtd = np.load(best_attack_path / 'attack_metrics.npz', allow_pickle=True)['rank_over_time']  # (16, T)
        loc_mtd = np.load(best_loc_path / 'attack_metrics.npz', allow_pickle=True)['rank_over_time']        # (16, T)
        traces_seen = np.arange(1, attack_mtd.shape[1] + 1)
        ax.plot(traces_seen, attack_mtd.max(axis=0), color='red', linewidth=lw, rasterized=True)
        ax.plot(traces_seen, loc_mtd.max(axis=0), color='blue', linewidth=lw, rasterized=True)
        ax.set_xlabel('Traces seen')
        ax.set_ylabel(r'Rank (worst byte) $\downarrow$')
        ax.set_title(fmt_dataset_name(dataset_id))
        ax.set_xscale('log')
    legend_handles = [
        Line2D([0], [0], color='red',  linewidth=lw, label='Best attacker (worst byte)'),
        Line2D([0], [0], color='blue', linewidth=lw, label='Best localizer (worst byte)'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncols=4, framealpha=0, bbox_to_anchor=(0.5, 0))
    fig.get_layout_engine().set(rect=(0, 0.15, 1, 1))
    fig.savefig(dest, dpi=DPI, bbox_inches='tight')
    plt.close(fig)

def run_plot_teaser_sweep(dest: Path):
    fig, ax = plt.subplots(1, 1, figsize=(0.4*WIDTH, 0.4*WIDTH), layout='constrained')
    sweep = load_sweep(get_output_dir('ascadv1-fixed') / 'htune_highdropout', 'ascadv1-fixed')
    acc = sweep['mean_acc']
    error = 1 - acc
    auroc = sweep[[f'white_box_auroc/input_x_gradient/{byte_idx}' for byte_idx in range(16)]].mean(axis=1)
    #auroc = sweep['white_box_auroc/input_x_gradient']
    ax.plot(error, auroc, marker='.', linestyle='none', markersize=3, color='purple', alpha=0.8, label='Single training run')
    oracle_val, random_val = _load_baseline_loc_metric('ascadv1-fixed', 'white_box_auroc/input_x_gradient')
    if oracle_val is not None:
        ax.axhline(oracle_val, color='green', linestyle=':', linewidth=1, label='Oracle performance')
    if random_val is not None:
        ax.axhline(random_val, color='grey',  linestyle=':', linewidth=1, label='Random guessing')
    ax.set_xlabel(r'Attack performance (avg. per-byte error) $\downarrow$', fontsize=6)
    ax.set_ylabel(r'Localization performance (AUROC w/ white-box) $\uparrow$', fontsize=6)
    ax.set_xscale('log')
    ax.legend(loc='lower left', framealpha=0, bbox_to_anchor=(0, 0.05))
    fig.savefig(dest, dpi=DPI)
    plt.close(fig)

def _try(fn):
    """Call fn(), return None on any exception."""
    try:
        return fn()
    except Exception:
        return None


def _wb_agreement_val(dataset_id: DATASET, leakiness: np.ndarray, key: str) -> Optional[float]:
    snr_dir = get_output_dir(dataset_id) / 'snr'
    if not snr_dir.exists():
        return None
    oracle = OracleAgreement(snr_dir, dataset_id)
    if key == 'spearman':
        return float(np.mean(oracle(leakiness)))
    else:
        return float(np.mean(oracle.get_auroc(leakiness)))


def _ta_mtd_mean(path: Path) -> float:
    d = np.load(path, allow_pickle=True)
    if 'ta-mtd/0' in d:
        return float(np.mean([float(d[f'ta-mtd/{b}']) for b in range(16)]))
    else:
        return float(d['mtd'].mean())


def _load_baseline_loc_metric(dataset_id: DATASET, metric: str) -> Tuple[Optional[float], Optional[float]]:
    """Return (oracle_val, random_val) for a given sweep column metric.
    Each value is None if its baseline file is missing."""
    baselines_dir = get_output_dir(dataset_id) / 'baselines'
    oracle_npy = baselines_dir / 'oracle.npy'
    random_npy = baselines_dir / 'random.npy'

    if metric.startswith('white_box_spearman'):
        key = 'spearman'
        o = _try(lambda: _wb_agreement_val(dataset_id, np.load(oracle_npy), key)) if oracle_npy.exists() else None
        r = _try(lambda: _wb_agreement_val(dataset_id, np.load(random_npy), key)) if random_npy.exists() else None
    elif metric.startswith('white_box_auroc'):
        key = 'auroc'
        o = _try(lambda: _wb_agreement_val(dataset_id, np.load(oracle_npy), key)) if oracle_npy.exists() else None
        r = _try(lambda: _wb_agreement_val(dataset_id, np.load(random_npy), key)) if random_npy.exists() else None
    elif metric.startswith('fwd_dnno'):
        o = _try(lambda: _load_occl(baselines_dir, 'fwd_dnno_occl.oracle', 'fwd-dnno-occl').mean())
        r = _try(lambda: _load_occl(baselines_dir, 'fwd_dnno_occl.random', 'fwd-dnno-occl').mean())
    elif metric.startswith('rev_dnno'):
        o = _try(lambda: _load_occl(baselines_dir, 'rev_dnno_occl.oracle', 'rev-dnno-occl').mean())
        r = _try(lambda: _load_occl(baselines_dir, 'rev_dnno_occl.random', 'rev-dnno-occl').mean())
    elif metric.startswith('ta_mtd'):
        oracle_path = baselines_dir / 'ta_mtd.oracle.npz'
        random_path = baselines_dir / 'ta_mtd.random.npz'
        o = _try(lambda: _ta_mtd_mean(oracle_path)) if oracle_path.exists() else None
        r = _try(lambda: _ta_mtd_mean(random_path)) if random_path.exists() else None
    else:
        return None, None
    return o, r


def run_plot_sweep(dest: Path):
    col_ylabels = [
        r'WB/Spearman $\uparrow$', r'WB/AUROC $\uparrow$', r'Fwd DNN occl. $\downarrow$',
        r'Rev DNN occl. $\uparrow$', r'TA MTD $\downarrow$',
    ]
    title_pad = 3
    fig, axes = plt.subplots(3, 5, sharex='row', layout='constrained', figsize=(WIDTH, 3*WIDTH/5))
    markersize = 2

    for dataset_id, axes_r in zip(['ascadv1-fixed', 'ascadv1-variable'], axes):
        sweep = load_sweep(get_output_dir(dataset_id) / 'htune_highdropout', dataset_id)
        best_attack_rv, best_loc_rv = get_best_runs(dataset_id)
        error = 1 - sweep['mean_acc']
        axes_r[0].set_xscale('log')
        axes_r[2].set_title(r'$\xleftarrow{\hspace{4em}}$ ' + fmt_dataset_name(dataset_id) + r' $\xrightarrow{\hspace{4em}}$', fontsize=6, pad=title_pad)
        for metric, ax, ylabel in zip([
            'white_box_spearman/input_x_gradient', 'white_box_auroc/input_x_gradient', 'fwd_dnno/input_x_gradient',
            'rev_dnno/input_x_gradient', 'ta_mtd/input_x_gradient'
        ], axes_r, col_ylabels):
            if 'dnno' not in metric:
                loc_metric = sweep[[f'{metric}/{byte_idx}' for byte_idx in range(16)]].mean(axis=1)
                best_attack_loc_metric = best_attack_rv[[f'{metric}/{byte_idx}' for byte_idx in range(16)]].mean()
                best_loc_loc_metric = best_loc_rv[[f'{metric}/{byte_idx}' for byte_idx in range(16)]].mean()
            else:
                loc_metric = sweep[metric]
                best_attack_loc_metric = best_attack_rv[metric]
                best_loc_loc_metric = best_loc_rv[metric]
            if 'ta_mtd' in metric:
                loc_metric = loc_metric.clip(upper=10_000)
                best_attack_loc_metric = min(best_attack_loc_metric, 10_000)
                best_loc_loc_metric = min(best_loc_loc_metric, 10_000)
            ax.plot(error, loc_metric, marker='.', linestyle='none', markersize=markersize/2, color='purple', alpha=0.8)
            ax.plot([1 - best_attack_rv['mean_acc']], [best_attack_loc_metric], color='red',  marker='*', markersize=3, label='Best attacker', zorder=5)
            ax.plot([1 - best_loc_rv['mean_acc']],    [best_loc_loc_metric],    color='blue', marker='*', markersize=3, label='Best localizer', zorder=5)
            oracle_val, random_val = _load_baseline_loc_metric(dataset_id, metric)
            if oracle_val is not None:
                ax.axhline(oracle_val, color='green', linestyle=':', linewidth=0.75, label='`Oracle\' (WB prof. set)')
            if random_val is not None:
                ax.axhline(random_val, color='grey',  linestyle=':', linewidth=0.75, label='Random baseline')
            ax.set_ylabel(ylabel, fontsize=5)
        for ax in axes_r:
            ax.set_xlabel(r'Error rate $\downarrow$', fontsize=5)

    # CHES-CTF-2018: no white-box metrics; use mean MTD as x-axis
    ches_sweep = load_sweep(get_output_dir('ches-ctf-2018') / 'htune_highdropout', 'ches-ctf-2018')
    mean_mtd = ches_sweep[[f'mtd/{byte_idx}' for byte_idx in range(16)]].mean(axis=1)
    best_attack_ches_rv, best_loc_ches_rv = get_best_runs('ches-ctf-2018')
    # Use axes[2][0] for legend; hide axes[2][1]
    axes[2][0].axis('off')
    axes[2][1].axis('off')
    axes[2][2].set_xscale('log')
    axes[2][2].set_title(fmt_dataset_name('ches-ctf-2018') + r' $\xrightarrow{\hspace{4em}}$', x=.85, fontsize=6, pad=title_pad)
    for metric, ax, ylabel in zip([
        'fwd_dnno/input_x_gradient', 'rev_dnno/input_x_gradient', 'ta_mtd/input_x_gradient'
    ], axes[2][2:], col_ylabels[2:]):
        if 'dnno' in metric:
            loc_metric = ches_sweep[metric]
            best_attack_loc_metric = best_attack_ches_rv[metric]
            best_loc_loc_metric = best_loc_ches_rv[metric]
        else:
            loc_metric = ches_sweep[[f'{metric}/{byte_idx}' for byte_idx in range(16)]].mean(axis=1)
            best_attack_loc_metric = best_attack_ches_rv[[f'{metric}/{byte_idx}' for byte_idx in range(16)]].mean()
            best_loc_loc_metric = best_loc_ches_rv[[f'{metric}/{byte_idx}' for byte_idx in range(16)]].mean()
        if 'ta_mtd' in metric:
            loc_metric = loc_metric.clip(upper=10_000)
            best_attack_loc_metric = min(best_attack_loc_metric, 10_000)
            best_loc_loc_metric = min(best_loc_loc_metric, 10_000)
        ax.plot(mean_mtd, loc_metric, marker='.', linestyle='none', markersize=markersize/2, color='purple', alpha=0.8)
        ax.plot([mean_mtd[best_attack_ches_rv.name]], [best_attack_loc_metric], color='red',  marker='*', markersize=3, label='Best attacker', zorder=5)
        ax.plot([mean_mtd[best_loc_ches_rv.name]],    [best_loc_loc_metric],    color='blue', marker='*', markersize=3, label='Best localizer', zorder=5)
        oracle_val, random_val = _load_baseline_loc_metric('ches-ctf-2018', metric)
        if oracle_val is not None:
            ax.axhline(oracle_val, color='green', linestyle=':', linewidth=0.75, label='Oracle')
        if random_val is not None:
            ax.axhline(random_val, color='grey',  linestyle=':', linewidth=0.75, label='Random')
        ax.set_ylabel(ylabel, fontsize=5)
        ax.set_xlabel(r'Mean MTD $\downarrow$', fontsize=5)

    # Legend in the blank axes[2][0]
    legend_handles = [
        Line2D([0], [0], color='purple', marker='.', linestyle='none', markersize=3, label='Tuning run'),
        Line2D([0], [0], color='red',    marker='*', linestyle='none', markersize=4, label='Best attacker'),
        Line2D([0], [0], color='blue',   marker='*', linestyle='none', markersize=4, label='Best localizer'),
        Line2D([0], [0], color='green',  linestyle=':', linewidth=0.75, label='`Oracle\' (WB on prof. set)'),
        Line2D([0], [0], color='grey',   linestyle=':', linewidth=0.75, label='Random guessing'),
    ]
    axes[2][1].legend(handles=legend_handles, loc='upper center', ncol=1, framealpha=0., fontsize=4)
    for row in axes:
        for ax in row[2:]:
            if ax.get_visible():
                ax.yaxis.set_major_formatter(plt.FuncFormatter(format_k))
    fig.savefig(dest, dpi=DPI)
    plt.close(fig)

def run_plot_dropout_sweep(dest: Path):
    """For each dataset, box-plot attack vs. localization performance
    binned by input dropout rate, using two y-axes."""
    # (dataset_id, atk_col, atk_label, atk_higher_better, loc_col, loc_label, loc_higher_better)
    configs = [
        ('ascadv1-fixed',
         'error_rate',                        r'Best attack perf. (avg. per-byte error) $\downarrow$',   False,
         'white_box_auroc/input_x_gradient',  r'Best loc. perf. (white-box AUROC) $\uparrow$',         True),
        ('ascadv1-variable',
         'error_rate',                        r'Best attack perf. (avg. per-byte error) $\downarrow$',   False,
         'white_box_auroc/input_x_gradient',  r'Best loc. perf. (white-box AUROC) $\uparrow$',         True),
        ('ches-ctf-2018',
         'per_byte_mtd',                     r'Best attack perf. (avg. per-byte MTD) $\downarrow$', False,
         'fwd_dnno/input_x_gradient',        r'Best loc. perf. (fwd DNN occl.) $\downarrow$',    False),
    ]
    bin_edges   = np.arange(0, 1.01, 0.1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    n_bins = len(bin_centers)
    atk_color = 'red'
    loc_color = 'blue'

    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, WIDTH/3), layout='constrained')
    twin_axes = []
    for ax, (dataset_id, atk_col, atk_label, atk_hi, loc_col, loc_label, loc_hi) in zip(axes, configs):
        sweep = load_sweep(get_output_dir(dataset_id) / 'htune_highdropout', dataset_id)

        dropout_rates = []
        for trial_path_str in sweep['path']:
            try:
                hparams = _load_swept_hparams(Path(trial_path_str))
                dropout_rates.append(hparams.get('model/input_dropout_rate', float('nan')))
            except Exception:
                dropout_rates.append(float('nan'))
        dropout = np.array(dropout_rates)

        if atk_col == 'per_byte_mtd':
            atk_vals = sweep[[f'mtd/{b}' for b in range(16)]].mean(axis=1).values
        elif atk_col == 'error_rate':
            atk_vals = 1 - sweep['mean_acc'].values
        else:
            atk_vals = sweep[atk_col].values

        if 'dnno' in loc_col:
            loc_vals = sweep[loc_col].values
        else:
            loc_vals = sweep[[f'{loc_col}/{b}' for b in range(16)]].mean(axis=1).values

        bin_idx = np.clip(np.digitize(dropout, bin_edges) - 1, 0, n_bins - 1)
        def _agg(vals, hi, fn):
            return np.array([fn(vals[bin_idx == i]) if (bin_idx == i).any() else np.nan for i in range(n_bins)])
        atk_bests = _agg(atk_vals, atk_hi, np.max if atk_hi else np.min)
        loc_bests = _agg(loc_vals, loc_hi, np.max if loc_hi else np.min)

        ax_loc = ax.twinx()
        twin_axes.append((ax, ax_loc, dataset_id))
        ax.plot(bin_centers, atk_bests, color=atk_color,  marker='o', markersize=2, linestyle=':', linewidth=0.2)
        ax_loc.plot(bin_centers, loc_bests, color=loc_color, marker='x', markersize=2, linestyle=':', linewidth=0.2)

        ax.set_yscale('log')
        if dataset_id == 'ches-ctf-2018':
            ax_loc.set_yscale('log')
        ax.set_xlabel(r'Input dropout rate', fontsize=6)
        ax.set_ylabel(atk_label, fontsize=6, color=atk_color)
        ax_loc.set_ylabel(loc_label, fontsize=6, color=loc_color)
        ax.set_xticks(bin_centers)
        ax.set_xticklabels([f'{c:.2f}' if i % 2 == 0 else '' for i, c in enumerate(bin_centers)], fontsize=5, rotation=45)
        ax.set_xlim(bin_edges[0] - 0.05, bin_edges[-1] + 0.05)
        ax.set_title(fmt_dataset_name(dataset_id), fontsize=7)

    legend_handles = [
        Line2D([0], [0], color=atk_color,  linestyle='-', linewidth=0.75, marker='o', markersize=2, label='Best attack'),
        Line2D([0], [0], color=loc_color, linestyle='-', linewidth=0.75, marker='o', markersize=2, label='Best localization'),
    ]
    # Force render so tick labels are created, then apply colors/sizes/rotation
    fig.canvas.draw()
    for ax, ax_loc, dataset_id in twin_axes:
        for lbl in ax.get_yticklabels(which='both'):
            lbl.set_color(atk_color)
            lbl.set_fontsize(3)
            lbl.set_rotation(45)
        for lbl in ax_loc.get_yticklabels(which='both'):
            lbl.set_color(loc_color)
            lbl.set_fontsize(3)
            lbl.set_rotation(45)
    fig.savefig(dest, dpi=DPI)
    plt.close(fig)


def run_plot_ta_mtd(dest: Path):
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, WIDTH/2.5))
    linewidth = 0.75
    traces_seen = np.arange(1, 10001)
    for dataset_id, ax in zip(['ascadv1-fixed', 'ascadv1-variable', 'ches-ctf-2018'], axes):
        best_attack_rv, best_loc_rv = get_best_runs(dataset_id)
        best_attack_path = Path(best_attack_rv['path'])
        best_loc_path = Path(best_loc_rv['path'])

        baselines_dir = get_output_dir(dataset_id) / 'baselines'
        random_path = baselines_dir / 'ta_mtd.random.npz'
        oracle_path = baselines_dir / 'ta_mtd.oracle.npz'
        if random_path.exists():
            random_rot = np.load(random_path, allow_pickle=True)['rank_over_time']
            ax.plot(traces_seen, np.mean(random_rot, axis=0), color='grey', linestyle='-', linewidth=linewidth, label='Random')
        if oracle_path.exists():
            oracle_rot = np.load(oracle_path, allow_pickle=True)['rank_over_time']
            ax.plot(traces_seen, np.mean(oracle_rot, axis=0), color='green', linestyle='-', linewidth=linewidth, label='White-box SNR')

        best_attack_rot = np.load(best_attack_path / 'ta_mtd.input_x_gradient.npz', allow_pickle=True)['rank_over_time']
        best_loc_rot    = np.load(best_loc_path    / 'ta_mtd.input_x_gradient.npz', allow_pickle=True)['rank_over_time']
        ax.plot(traces_seen, np.mean(best_attack_rot, axis=0), color='red',  linestyle='-', linewidth=linewidth, label='Best attacker')
        ax.plot(traces_seen, np.mean(best_loc_rot,    axis=0), color='blue', linestyle='-', linewidth=linewidth, label='Best localizer')

        ax.set_xlabel('Traces seen')
        ax.set_ylabel('Rank (avg. per byte)')
        ax.set_title(fmt_dataset_name(dataset_id))
        ax.set_xscale('log')
    seen, handles, labels = set(), [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l)
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc='lower center', ncols=4, framealpha=0, bbox_to_anchor=(0.5, 0))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.3)
    fig.savefig(dest, dpi=DPI)
    plt.close(fig)

def _load_occl(base: Path, stem: str, npz_key: str) -> np.ndarray:
    """Load a 1-D occlusion curve from .npz (new) or .npy (old) format.

    .npz files store two arrays: npz_key (mean over all bytes) and npz_key+'/2'
    (byte-2 only). .npy files store only the mean as a plain array.
    """
    npz = base / f'{stem}.npz'
    npy = base / f'{stem}.npy'
    if npz.exists():
        return np.load(npz, allow_pickle=True)[npz_key]
    elif npy.exists():
        return np.load(npy)
    else:
        raise FileNotFoundError(f'Neither {npz} nor {npy} exists')


def run_plot_dnn_occlusion(dest: Path):
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, WIDTH/2.5))
    linewidth = 0.75
    for dataset_id, ax in zip(['ascadv1-fixed', 'ascadv1-variable', 'ches-ctf-2018'], axes):
        best_attack_rv, best_loc_rv = get_best_runs(dataset_id)
        best_attack_path = Path(best_attack_rv['path'])
        best_loc_path = Path(best_loc_rv['path'])
        feature_count = FEATURE_COUNTS[dataset_id]
        present_features = np.linspace(0, feature_count, 101)[:-1]

        baselines_dir = get_output_dir(dataset_id) / 'baselines'
        has_random = (baselines_dir / 'fwd_dnno_occl.random.npz').exists() or (baselines_dir / 'fwd_dnno_occl.random.npy').exists()
        has_oracle = (baselines_dir / 'fwd_dnno_occl.oracle.npz').exists() or (baselines_dir / 'fwd_dnno_occl.oracle.npy').exists()
        if has_random:
            ax.plot(present_features, _load_occl(baselines_dir, 'fwd_dnno_occl.random', 'fwd-dnno-occl'), color='grey', linestyle=':', linewidth=linewidth, label='Random (forward)')
            ax.plot(present_features, _load_occl(baselines_dir, 'rev_dnno_occl.random', 'rev-dnno-occl'), color='grey', linestyle='--', linewidth=linewidth, label='Random (reverse)')
        if has_oracle:
            ax.plot(present_features, _load_occl(baselines_dir, 'fwd_dnno_occl.oracle', 'fwd-dnno-occl'), color='green', linestyle=':', linewidth=linewidth, label='White-box SNR (forward)')
            ax.plot(present_features, _load_occl(baselines_dir, 'rev_dnno_occl.oracle', 'rev-dnno-occl'), color='green', linestyle='--', linewidth=linewidth, label='White-box SNR (reverse)')

        best_attack_fwd = _load_occl(best_attack_path, 'fwd_dnno_occl.input_x_gradient', 'fwd-dnno-occl')
        best_attack_rev = _load_occl(best_attack_path, 'rev_dnno_occl.input_x_gradient', 'rev-dnno-occl')
        best_loc_fwd    = _load_occl(best_loc_path,    'fwd_dnno_occl.input_x_gradient', 'fwd-dnno-occl')
        best_loc_rev    = _load_occl(best_loc_path,    'rev_dnno_occl.input_x_gradient', 'rev-dnno-occl')
        ax.plot(present_features, best_attack_fwd, color='red',  linestyle=':', linewidth=linewidth, label='Best attacker (forward)')
        ax.plot(present_features, best_attack_rev, color='red',  linestyle='--', linewidth=linewidth, label='Best attacker (reverse)')
        ax.plot(present_features, best_loc_fwd,    color='blue', linestyle=':', linewidth=linewidth, label='Best localizer (forward)')
        ax.plot(present_features, best_loc_rev,    color='blue', linestyle='--', linewidth=linewidth, label='Best localizer (reverse)')

        ax.set_xlabel('Included features')
        ax.set_ylabel('MTD of attacker (avg. per byte)')
        ax.set_title(fmt_dataset_name(dataset_id))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(-2, 2), useMathText=True)
        ax.set_yscale('log')
    # Gather all unique legend entries across axes
    seen, handles, labels = set(), [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l)
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc='lower center', ncols=4, framealpha=0, bbox_to_anchor=(0.5, 0))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.3)
    fig.savefig(dest, dpi=DPI)
    plt.close(fig)

def run_plot_oracle_agreement(dest: Path, dataset_id: Literal['ascadv1-fixed', 'ascadv1-variable'] = 'ascadv1-variable'):
    best_attack_rv, best_loc_rv = get_best_runs(dataset_id)
    best_attack_path = Path(best_attack_rv['path'])
    best_attack_auroc = best_attack_rv['white_box_auroc/input_x_gradient/2']
    best_loc_path = Path(best_loc_rv['path'])
    best_loc_auroc = best_loc_rv['white_box_auroc/input_x_gradient/2']
    best_attack_inputxgrad = np.load(best_attack_path / 'input_x_gradient.npy')[2, :]
    best_loc_inputxgrad = np.load(best_loc_path / 'input_x_gradient.npy')[2, :]
    title_pad = 3
    h_pad = 1/72  # inches; default is 4/72
    fig = plt.figure(figsize=(WIDTH, WIDTH/2), constrained_layout=True)
    fig.get_layout_engine().set(h_pad=h_pad, w_pad=1/72, hspace=0.05, wspace=0.05)
    time_fig, scatter_fig = fig.subfigures(1, 2, wspace=0.05)
    time_axes = time_fig.subplots(3, 1, sharex=True)
    scatter_axes = scatter_fig.subplot_mosaic(
        [['comp',       'r_in',   'r2'   ],
         ['r_out',      'S2xr2',  'Srout'],
         ['k2w2rin', 'k2w2r2', 'marginals']],
        sharex=True, sharey=True,
    )
    time_axes[2].set_xlabel(r'Time $t$')
    time_axes[1].set_ylabel(r'Leakiness of $X_t$')
    time_axes[0].set_title(r'Input $*$ Grad (best attacker)', fontsize=7, pad=title_pad)
    time_axes[1].set_title(r'Input $*$ Grad (best localizer)', fontsize=7, pad=title_pad)
    time_axes[2].set_title(r'White-box SNR', fontsize=7, pad=title_pad)
    time_axes[0].plot(best_attack_inputxgrad, rasterized=True, linewidth=0.1, marker='.', markersize=1, color='red')
    time_axes[1].plot(best_loc_inputxgrad, rasterized=True, linewidth=0.1, marker='.', markersize=1, color='blue')
    white_box_snrs = plot_ascadv1_oracle_leakiness(get_output_dir(dataset_id) / 'snr', time_axes[2])
    best_attacker_spearman = spearmanr(white_box_snrs['composite'], best_attack_inputxgrad).statistic
    best_localizer_spearman = spearmanr(white_box_snrs['composite'], best_loc_inputxgrad).statistic
    time_axes[0].text(
        0.01, 0.95, r"Spearman's $\rho$ w/ white-box SNR: " + f"{best_attacker_spearman:.3f}",
        transform=time_axes[0].transAxes, ha='left', va='top', fontsize=4,
    )
    time_axes[0].text(
        0.01, 0.85, r"AUROC w/ white-box SNR: " + f"{best_attack_auroc:.3f}",
        transform=time_axes[0].transAxes, ha='left', va='top', fontsize=4,
    )
    time_axes[1].text(
        0.99, 0.95, r"Spearman's $\rho$ w/ white-box SNR: " + f"{best_localizer_spearman:.3f}",
        transform=time_axes[1].transAxes, ha='right', va='top', fontsize=4,
    )
    time_axes[1].text(
        0.99, 0.85, r"AUROC w/ white-box SNR: " + f"{best_loc_auroc:.3f}",
        transform=time_axes[1].transAxes, ha='right', va='top', fontsize=4,
    )
    white_box_snrs['pr'] -= white_box_snrs['pr'].min()
    white_box_snrs['pr'] += white_box_snrs['prin'].min()
    time_axes[2].legend(loc='upper right', ncol=3, framealpha=0, fontsize=3.9, labelspacing=0.2, columnspacing=1.0, handlelength=1.0)
    for ax in time_axes:
        ax.ticklabel_format(style='sci', axis='x', scilimits=(-2, 2), useMathText=True)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(-2, 2), useMathText=True)
    for ax in scatter_axes.values():
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(axis='both', which='both', labelsize=4)
    scatter_axes['k2w2r2'].set_xlabel(r'White box SNR')
    scatter_axes['r_out'].set_ylabel(r'Input $*$ Grad')
    scatter_axes['comp'].set_title(r'Avg. of all', fontsize=7, pad=title_pad)
    scatter_axes['r_in'].set_title(r'$r_{\mathrm{in}}$', fontsize=7, pad=title_pad)
    scatter_axes['r2'].set_title(r'$r_2$', fontsize=7, pad=title_pad)
    scatter_axes['r_out'].set_title(r'$r_{\mathrm{out}}$', fontsize=7, pad=title_pad)
    scatter_axes['S2xr2'].set_title(r'$S_2 \oplus r_2$', fontsize=7, pad=title_pad)
    scatter_axes['Srout'].set_title(r'$S_r \oplus r_{\mathrm{out}}$', fontsize=7, pad=title_pad)
    scatter_axes['k2w2rin'].set_title(r'$k_2 \oplus w_2 \oplus r_{\mathrm{in}}$', fontsize=7, pad=title_pad)
    scatter_axes['k2w2r2'].set_title(r'$k_2 \oplus w_2 \oplus r_2$', fontsize=7, pad=title_pad)
    loc_scatter_kwargs    = dict(color='blue', linestyle='none', marker='.', markersize=0.5, alpha=0.2, rasterized=True)
    attack_scatter_kwargs = dict(color='red',  linestyle='none', marker='+', markersize=0.5, alpha=0.2, rasterized=True)
    snr_ixg_pairs = [
        ('comp',    'composite'),
        ('r_in',    'rin'),
        ('r2',      'r'),
        ('r_out',   'rout'),
        ('S2xr2',   'yr'),
        ('Srout',   'yrout'),
        ('k2w2rin', 'prin'),
        ('k2w2r2',  'pr'),
    ]
    for ax_key, snr_key in snr_ixg_pairs:
        scatter_axes[ax_key].plot(white_box_snrs[snr_key], best_attack_inputxgrad, **attack_scatter_kwargs)
        scatter_axes[ax_key].plot(white_box_snrs[snr_key], best_loc_inputxgrad,    **loc_scatter_kwargs)

    # Draw a horizontal dotted line at the KDE mode of each model's IxG distribution.
    _ixg_modes = {}
    for _ixg_vals, _color in [(best_attack_inputxgrad, 'red'), (best_loc_inputxgrad, 'blue')]:
        _ixg_pos = _ixg_vals[_ixg_vals > 0]
        _ixg_log = np.log10(_ixg_pos)
        _eval_grid = np.linspace(_ixg_log.min(), _ixg_log.max(), 1000)
        _kde_vals = gaussian_kde(_ixg_log)(_eval_grid)
        _mode = 10 ** _eval_grid[np.argmax(_kde_vals)]
        _ixg_modes[_color] = _mode
        for ax_key, _ in snr_ixg_pairs:
            scatter_axes[ax_key].axhline(_mode, color=_color, linestyle=':', linewidth=0.4, alpha=0.9, zorder=4)

    marg_ax = scatter_axes['marginals']
    marg_ax.set_title(r'Densities', fontsize=7, pad=title_pad)
    snr_vals = white_box_snrs['composite']
    snr_log  = np.log10(snr_vals[snr_vals > 0])
    for ixg_vals, color, label in [
        (best_attack_inputxgrad, 'red',  r'Best attacker'),
        (best_loc_inputxgrad,    'blue', r'Best localizer'),
    ]:
        ixg_log     = np.log10(ixg_vals[ixg_vals > 0])
        ixg_grid    = np.linspace(ixg_log.min(), ixg_log.max(), 300)
        ixg_density = gaussian_kde(ixg_log)(ixg_grid)
        ixg_density_scaled = 10 ** (snr_log.min() + (ixg_density / ixg_density.max()) * (snr_log.max() - snr_log.min()))
        marg_ax.plot(ixg_density_scaled, 10**ixg_grid, color=color, linewidth=0.5, label=label)
    for _color, _mode in _ixg_modes.items():
        marg_ax.axhline(_mode, color=_color, linestyle=':', linewidth=0.4, alpha=0.9, zorder=4)
    marg_ax.tick_params(axis='y', which='both', labelleft=False)
    #marg_ax.legend(loc='upper right', fontsize=4, labelspacing=0.2, handlelength=1.0, framealpha=0)
    fig.savefig(dest, dpi=DPI)
    plt.close(fig)

def run_plot_per_byte_leakiness(dest: Path):
    """5-row x 16-col grid of per-byte leakiness curves.

    Rows 0/1: ASCADv1-fixed  black-box / white-box
    Rows 2/3: ASCADv1-variable black-box / white-box
    Row  4:   CHES-CTF-2018 black-box only
    """
    row_specs = [
        ('ascadv1-fixed',    'black_box'),
        ('ascadv1-fixed',    'white_box'),
        ('ascadv1-variable', 'black_box'),
        ('ascadv1-variable', 'white_box'),
        ('ches-ctf-2018',    'black_box'),
    ]
    # one label per group of rows; rows listed top-to-bottom
    group_specs = [
        ([0, 1], 'ASCADv1 (fixed)'),
        ([2, 3], 'ASCADv1 (variable)'),
        ([4],    'CC18'),
    ]
    n_rows, n_cols = len(row_specs), 16
    lw = 0.3
    spine_lw = 0.3
    gap_after = {1, 3}
    gap_size = WIDTH / n_cols * 0.15  # narrow gap between dataset groups

    ax_size = WIDTH / n_cols  # square axes, in inches
    fig_height = n_rows * ax_size + len(gap_after) * gap_size
    fig = plt.figure(figsize=(WIDTH, fig_height))

    ax_w = 1.0 / n_cols
    ax_h = ax_size / fig_height
    gap_h = gap_size / fig_height

    # Compute top/bottom of each row in figure coordinates
    row_tops    = []
    row_bottoms = []
    cumulative_gap = 0.0
    for row_idx in range(n_rows):
        top    = 1.0 - row_idx * ax_h - cumulative_gap
        bottom = top - ax_h
        row_tops.append(top)
        row_bottoms.append(bottom)
        if row_idx in gap_after:
            cumulative_gap += gap_h

    axes = np.empty((n_rows, n_cols), dtype=object)
    for row_idx in range(n_rows):
        for byte_idx in range(n_cols):
            ax = fig.add_axes([byte_idx * ax_w, row_bottoms[row_idx], ax_w, ax_h])
            axes[row_idx, byte_idx] = ax

    loc_ixg = {}
    for dataset_id in ['ascadv1-fixed', 'ascadv1-variable', 'ches-ctf-2018']:
        _, best_loc_rv = get_best_runs(dataset_id)
        loc_ixg[dataset_id] = np.load(Path(best_loc_rv['path']) / 'input_x_gradient.npy')

    for row_idx, (dataset_id, kind) in enumerate(row_specs):
        for byte_idx in range(n_cols):
            ax = axes[row_idx, byte_idx]
            if kind == 'black_box':
                ax.plot(loc_ixg[dataset_id][byte_idx], linewidth=lw, color='blue', rasterized=True)
            else:
                plot_ascadv1_oracle_leakiness(get_output_dir(dataset_id) / 'snr', ax, byte=byte_idx, markers=False)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(spine_lw)
            if row_idx == 0:
                ax.set_title(f'Byte {byte_idx}', fontsize=6, pad=2)

    # x-label only on bottom row, centered
    fig.text(0.5, row_bottoms[-1] - 0.01, r'$\xleftarrow{\hspace{4em}}$ Time $t$ $\xrightarrow{\hspace{4em}}$', ha='center', va='top', fontsize=6)

    # Shared "Estimated leakiness" label centered over all rows
    all_center_y = (row_tops[0] + row_bottoms[-1]) / 2
    fig.text(-0.03, all_center_y, r'$\xleftarrow{\hspace{4em}}$ Estimated leakiness of $X_t$ $\xrightarrow{\hspace{4em}}$',
             rotation=90, va='center', ha='center', fontsize=6)

    # Per-group dataset name labels, just to the right of "Estimated leakiness"
    for group_rows, dataset_label in group_specs:
        center_y = (row_tops[group_rows[0]] + row_bottoms[group_rows[-1]]) / 2
        fig.text(-0.01, center_y, dataset_label,
                 rotation=90, va='center', ha='center', fontsize=5)

    # Legend: blue = black-box, then oracle intermediate variable colors from a typical byte
    legend_lw = lw * 3
    bb_handle = Line2D([0], [0], color='blue', linewidth=legend_lw, label=r'\textbf{Ours}: Input $*$ Grad (best localizer)')
    # Get white-box handles from a masked byte (byte 2) and the unmasked case (byte 0)
    dummy_fig, dummy_ax = plt.subplots()
    plot_ascadv1_oracle_leakiness(get_output_dir('ascadv1-fixed') / 'snr', dummy_ax, byte=2, markers=False, arb_byte=True)
    wb_handles, wb_labels = dummy_ax.get_legend_handles_labels()
    grey_ax = dummy_fig.add_subplot()
    plot_ascadv1_oracle_leakiness(get_output_dir('ascadv1-fixed') / 'snr', grey_ax, byte=0, markers=False, arb_byte=True)
    grey_handles, grey_labels = grey_ax.get_legend_handles_labels()
    plt.close(dummy_fig)
    for h in grey_handles + wb_handles:
        h.set_linewidth(legend_lw)
    all_handles = [bb_handle] + grey_handles + wb_handles
    fig.legend(handles=all_handles, loc='lower center', ncols=ceil(len(all_handles)/2),
               framealpha=0, fontsize=5, bbox_to_anchor=(0.5, -.225),
               handlelength=1.5, columnspacing=2)

    fig.savefig(dest, dpi=DPI, bbox_inches='tight')
    plt.close(fig)

def _load_swept_hparams(trial_path: Path) -> dict:
    with open(trial_path / 'config.yaml') as f:
        cfg = yaml.safe_load(f)
    with open(trial_path / 'hparams.yaml') as f:
        raw = re.sub(r'!!python/\S+', '', f.read())
    hparams = yaml.safe_load(raw)
    search_space = cfg.get('search_space', {})
    result = {}
    for section, params in search_space.items():
        for param_name in params:
            if section == 'model':
                value = hparams['model_kwargs'][param_name]
            else:  # training and other top-level sections are flattened
                value = hparams[param_name]
            result[f'{section}/{param_name}'] = value
    return result

def _print_perf_rows(label: str, rv: pandas.Series, include_acc: bool = True):
    mtd_vals  = [rv['mtd']]  + [rv[f'mtd/{b}']  for b in range(16)]
    mtd_str   = ' & '.join(f'{v:.3f}' for v in mtd_vals)
    print(f'  MTD  | {label} & {mtd_str}')
    if include_acc:
        acc_vals = [rv['acc']] + [rv[f'acc/{b}'] for b in range(16)]
        acc_str  = ' & '.join(f'{v:.3f}' for v in acc_vals)
        print(f'  Acc  | {label} & {acc_str}')

def run_print_best_hparams():
    for dataset_id in ['ascadv1-fixed', 'ascadv1-variable', 'ches-ctf-2018']:
        print(f'\n=== {fmt_dataset_name(dataset_id)} ===')
        best_attack_rv, best_loc_rv = get_best_runs(dataset_id)
        best_attack_path = Path(best_attack_rv['path'])
        best_loc_path = Path(best_loc_rv['path'])

        if dataset_id == 'ches-ctf-2018':
            print(f'  Best attacker (mean MTD = {best_attack_rv["mtd"]:.3f}):')
        else:
            print(f'  Best attacker (mean_acc = {best_attack_rv["mean_acc"]:.4f}):')
        for k, v in _load_swept_hparams(best_attack_path).items():
            print(f'    {k}: {v}')
        _print_perf_rows('Best attacker', best_attack_rv)

        if dataset_id == 'ches-ctf-2018':
            print(f'  Best localizer (avg rank across fwd_dnno, rev_dnno, ta_mtd):')
        else:
            print(f'  Best localizer (mean white-box AUROC = {best_loc_rv["white_box_auroc/input_x_gradient"]:.4f}):')
        for k, v in _load_swept_hparams(best_loc_path).items():
            print(f'    {k}: {v}')
        _print_perf_rows('Best localizer', best_loc_rv)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--plot-everything', default=False, action='store_true'
    )
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
        '--plot-oracle-agreement', default=False, action='store_true'
    )
    parser.add_argument(
        '--plot-dnn-occlusion', default=False, action='store_true'
    )
    parser.add_argument(
        '--plot-ta-mtd', default=False, action='store_true'
    )
    parser.add_argument(
        '--plot-sweep', default=False, action='store_true'
    )
    parser.add_argument(
        '--plot-dropout-sweep', default=False, action='store_true'
    )
    parser.add_argument(
        '--plot-per-byte-leakiness', default=False, action='store_true'
    )
    parser.add_argument(
        '--plot-teaser', default=False, action='store_true'
    )
    parser.add_argument(
        '--print-best-hparams', default=False, action='store_true'
    )
    parser.add_argument(
        '--dest', default=None, type=Path
    )
    args = parser.parse_args()

    plot_everything: bool = args.plot_everything
    assert isinstance(plot_everything, bool)
    plot_training_curves: bool = args.plot_training_curves
    assert isinstance(plot_training_curves, bool)
    plot_mtd_curves: bool = args.plot_mtd_curves
    assert isinstance(plot_mtd_curves, bool)
    format_attack_performance: bool = args.format_attack_performance
    assert isinstance(format_attack_performance, bool)
    plot_cost_scaling: bool = args.plot_cost_scaling
    assert isinstance(plot_cost_scaling, bool)
    plot_oracle_agreement: bool = args.plot_oracle_agreement
    assert isinstance(plot_oracle_agreement, bool)
    plot_dnn_occlusion: bool = args.plot_dnn_occlusion
    assert isinstance(plot_dnn_occlusion, bool)
    plot_ta_mtd: bool = args.plot_ta_mtd
    assert isinstance(plot_ta_mtd, bool)
    plot_sweep: bool = args.plot_sweep
    assert isinstance(plot_sweep, bool)
    plot_dropout_sweep: bool = args.plot_dropout_sweep
    assert isinstance(plot_dropout_sweep, bool)
    plot_teaser: bool = args.plot_teaser
    assert isinstance(plot_teaser, bool)
    plot_per_byte_leakiness: bool = args.plot_per_byte_leakiness
    assert isinstance(plot_per_byte_leakiness, bool)
    print_best_hparams: bool = args.print_best_hparams
    assert isinstance(print_best_hparams, bool)
    dest: Optional[Path] = args.dest
    if dest is None:
        dest = OUTPUTS_ROOT / 'plots_for_paper'
    assert isinstance(dest, Path)
    dest.mkdir(exist_ok=True, parents=True)

    if plot_training_curves or plot_everything:
        run_plot_training_curves(dest / 'training_curves.pdf')
    if plot_mtd_curves or plot_everything:
        run_plot_mtd_curves(dest / 'mtd_curves.pdf')
    if plot_cost_scaling or plot_everything:
        run_plot_cost_scaling(dest / 'cost_scaling.pdf')
    if plot_oracle_agreement or plot_everything:
        for _dataset_id in ['ascadv1-fixed', 'ascadv1-variable']:
            run_plot_oracle_agreement(dest / f'oracle_agreement_{_dataset_id}.pdf', dataset_id=_dataset_id)
    if plot_dnn_occlusion or plot_everything:
        run_plot_dnn_occlusion(dest / 'dnn_occlusion.pdf')
    if plot_ta_mtd or plot_everything:
        run_plot_ta_mtd(dest / 'ta_mtd.pdf')
    if plot_sweep or plot_everything:
        run_plot_sweep(dest / 'sweep.pdf')
    if plot_dropout_sweep or plot_everything:
        run_plot_dropout_sweep(dest / 'dropout_sweep.pdf')
    if plot_teaser or plot_everything:
        run_plot_teaser_sweep(dest / 'teaser_sweep.pdf')
    if plot_per_byte_leakiness or plot_everything:
        run_plot_per_byte_leakiness(dest / 'per_byte_leakiness.pdf')
    if print_best_hparams or plot_everything:
        run_print_best_hparams()

if __name__ == '__main__':
    main()