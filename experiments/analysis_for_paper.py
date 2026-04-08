import argparse
from pathlib import Path
from typing import Optional, Literal, List, get_args
from collections import defaultdict
from tqdm import tqdm

import pandas
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from leakage_localization.datasets import DATASET, PARTITION
from leakage_localization.training.parse_metrics import parse_metrics
from leakage_localization.evaluation import OracleAgreement

from init_things import *
from utils.visualize_runs import *

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
                oracle_agreement = OracleAgreement(
                    get_output_dir(dataset_id) / 'snr', dataset_id
                )
                leakiness_estimates = np.load(trial_dir / f'{attr_method}.npy')
                data[f'white_box_spearman/{attr_method}'].append(oracle_agreement.get_full_spearman(leakiness_estimates))
                data[f'white_box_auroc/{attr_method}'].append(oracle_agreement.get_full_auroc(leakiness_estimates))
                per_byte_spearman = oracle_agreement(leakiness_estimates)
                per_byte_auroc = oracle_agreement.get_auroc(leakiness_estimates)
                for byte_idx in range(16):
                    data[f'white_box_spearman/{attr_method}/{byte_idx}'].append(per_byte_spearman[byte_idx])
                    data[f'white_box_auroc/{attr_method}/{byte_idx}'].append(per_byte_auroc[byte_idx])
        data = pandas.DataFrame(data)
        data['mean_acc'] = data[[f'acc/{byte_idx}' for byte_idx in range(16)]].mean(axis=1)
        for attr_method in ['gradvis', 'input_x_gradient']:
            data[f'mean_ta_mtd/{attr_method}'] = data[[f'ta_mtd/{attr_method}/{byte_idx}' for byte_idx in range(16)]].mean(axis=1)
        data.to_csv(sweep_dir / 'sweep_summary.csv')
    data = pandas.read_csv(sweep_dir / 'sweep_summary.csv')
    return data

def run_plot_oracle_agreement(dest: Path, dataset_id: Literal['ascadv1-fixed', 'ascadv1-variable'] = 'ascadv1-fixed'):
    sweep_path = get_output_dir(dataset_id) / 'htune_highdropout'
    sweep = load_sweep(sweep_path, dataset_id)
    best_attack_idx = sweep['acc'].idxmax()
    best_attack_auroc = sweep.loc[best_attack_idx]['white_box_auroc/input_x_gradient/2']
    best_attack_path = Path(sweep.loc[best_attack_idx]['path'])
    best_attack_inputxgrad = np.load(best_attack_path / 'input_x_gradient.npy')[2, :]
    best_loc_idx = sweep['white_box_auroc/input_x_gradient'].idxmax()
    best_loc_path = Path(sweep.loc[best_loc_idx]['path'])
    best_loc_auroc = sweep.loc[best_loc_idx]['white_box_auroc/input_x_gradient/2']
    best_loc_inputxgrad = np.load(best_loc_path / 'input_x_gradient.npy')[2, :]
    title_pad = 3
    h_pad = 1/72  # inches; default is 4/72
    fig = plt.figure(figsize=(WIDTH, WIDTH/2), constrained_layout=True)
    fig.get_layout_engine().set(h_pad=h_pad)
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
    time_axes[2].legend(loc='upper right', ncol=3, framealpha=0, fontsize=4, labelspacing=0.2, columnspacing=2.0, handlelength=1.0)
    for ax in time_axes:
        ax.ticklabel_format(style='sci', axis='x', scilimits=(-2, 2), useMathText=True)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(-2, 2), useMathText=True)
    for ax in scatter_axes.values():
        ax.set_xscale('log')
        ax.set_yscale('log')
    scatter_axes['k2w2r2'].set_xlabel(r'White box SNR')
    scatter_axes['r_out'].set_ylabel(r'Input $*$ Grad (best localizer)')
    scatter_axes['comp'].set_title(r'Avg. of all', fontsize=7, pad=title_pad)
    scatter_axes['r_in'].set_title(r'$r_{\mathrm{in}}$', fontsize=7, pad=title_pad)
    scatter_axes['r2'].set_title(r'$r_2$', fontsize=7, pad=title_pad)
    scatter_axes['r_out'].set_title(r'$r_{\mathrm{out}}$', fontsize=7, pad=title_pad)
    scatter_axes['S2xr2'].set_title(r'$S_2 \oplus r_2$', fontsize=7, pad=title_pad)
    scatter_axes['Srout'].set_title(r'$S_r \oplus r_{\mathrm{out}}$', fontsize=7, pad=title_pad)
    scatter_axes['k2w2rin'].set_title(r'$k_2 \oplus w_2 \oplus r_{\mathrm{in}}$', fontsize=7, pad=title_pad)
    scatter_axes['k2w2r2'].set_title(r'$k_2 \oplus w_2 \oplus r_2$', fontsize=7, pad=title_pad)
    scatter_kwargs = dict(color='blue', linestyle='none', marker='.', markersize=1, alpha=0.2, rasterized=True)
    scatter_axes['comp'].plot(white_box_snrs['composite'], best_loc_inputxgrad, **scatter_kwargs)
    scatter_axes['r_in'].plot(white_box_snrs['rin'], best_loc_inputxgrad, **scatter_kwargs)
    scatter_axes['r2'].plot(white_box_snrs['r'], best_loc_inputxgrad, **scatter_kwargs)
    scatter_axes['r_out'].plot(white_box_snrs['rout'], best_loc_inputxgrad, **scatter_kwargs)
    scatter_axes['S2xr2'].plot(white_box_snrs['yr'], best_loc_inputxgrad, **scatter_kwargs)
    scatter_axes['Srout'].plot(white_box_snrs['yrout'], best_loc_inputxgrad, **scatter_kwargs)
    scatter_axes['k2w2rin'].plot(white_box_snrs['prin'], best_loc_inputxgrad, **scatter_kwargs)
    scatter_axes['k2w2r2'].plot(white_box_snrs['pr'], best_loc_inputxgrad, **scatter_kwargs)
    snr_color = 'green'
    ixg_color = 'orange'
    marg_ax = scatter_axes['marginals']
    marg_ax.spines['left'].set_color(ixg_color)
    marg_ax.spines['bottom'].set_color(snr_color)
    marg_ax.tick_params(axis='y', colors=ixg_color)
    marg_ax.tick_params(axis='x', which='both', color=snr_color, labelcolor='black')
    snr_vals = white_box_snrs['composite']
    ixg_vals = best_loc_inputxgrad
    snr_log = np.log10(snr_vals[snr_vals > 0])
    ixg_log = np.log10(ixg_vals[ixg_vals > 0])
    snr_grid = np.linspace(snr_log.min(), snr_log.max(), 300)
    ixg_grid = np.linspace(ixg_log.min(), ixg_log.max(), 300)
    snr_density = gaussian_kde(snr_log)(snr_grid)
    ixg_density = gaussian_kde(ixg_log)(ixg_grid)
    # Scale density to the range of the other axis so shared limits are not expanded
    snr_density_scaled = 10 ** (ixg_log.min() + (snr_density / snr_density.max()) * (ixg_log.max() - ixg_log.min()))
    ixg_density_scaled = 10 ** (snr_log.min() + (ixg_density / ixg_density.max()) * (snr_log.max() - snr_log.min()))
    marg_ax.plot(10**snr_grid, snr_density_scaled, color=snr_color, linewidth=0.5, label=r'White-box SNR')
    marg_ax.plot(ixg_density_scaled, 10**ixg_grid, color=ixg_color, linewidth=0.5, label=r'Input $*$ Grad')
    marg_ax.legend(loc='upper right', fontsize=4, labelspacing=0.2, columnspacing=2.0, handlelength=1.0, framealpha=0)
    marg_ax.set_title(r'Densities', fontsize=7, pad=title_pad)
    fig.savefig(dest, dpi=DPI)
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
        '--plot-oracle-agreement', default=False, action='store_true'
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
    plot_oracle_agreement: bool = args.plot_oracle_agreement
    assert isinstance(plot_oracle_agreement, bool)
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
    if plot_oracle_agreement:
        run_plot_oracle_agreement(dest / 'oracle_agreement.pdf')

if __name__ == '__main__':
    main()