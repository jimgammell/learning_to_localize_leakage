"""Temporary script: regularization sweep analysis with classification + oracle agreement.
Files created: experiments/_tmp_reg_sweep_*.png — delete when done.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from numpy.typing import NDArray
from matplotlib import pyplot as plt
import yaml

from init_things import *
from leakage_localization.evaluation import OracleAgreement

NUM_BYTES = 16
DATASET_ID = 'ascadv1-fixed'
DATASET_KEY = 'ascadv1_fixed'
BASE_DIR = OUTPUTS_ROOT / DATASET_KEY / 'reg_sweep'
SNR_DIR = OUTPUTS_ROOT / DATASET_KEY / 'snr'

SWEEP_PREFIXES = [
    'input_dropout', 'hidden_dropout', 'weight_decay',
    'gaussian_noise', 'random_roll', 'random_lpf', 'mixup',
]
LOG_SCALE_PREFIXES = {'weight_decay', 'gaussian_noise'}

SWEEP_PREFIX_TO_YAML_KEY = {
    'input_dropout':  ('model', 'input_dropout_rate'),
    'hidden_dropout': ('model', 'hidden_dropout_rate'),
    'weight_decay':   ('training', 'weight_decay'),
    'gaussian_noise': ('training', 'additive_gaussian_noise'),
    'random_roll':    ('data', 'random_roll_scale'),
    'random_lpf':     ('data', 'random_lpf_scale'),
    'mixup':          ('training', 'mixup_alpha'),
}


def load_defaults() -> Dict[str, float]:
    config_path = Path(f'./config/{DATASET_KEY}.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return {prefix: float(config[section][key]) for prefix, (section, key) in SWEEP_PREFIX_TO_YAML_KEY.items()}


def load_all_runs():
    """Load classification metrics and compute oracle agreement for every run.
    Returns dict: {prefix: [(reg_value, seed, per_byte_acc[16], full_acc, per_byte_oa[16]), ...]}
    """
    oracle_metric = OracleAgreement(SNR_DIR, DATASET_ID)

    all_data = {p: [] for p in SWEEP_PREFIXES}
    for prefix in SWEEP_PREFIXES:
        for d in sorted(BASE_DIR.iterdir()):
            if not d.is_dir() or not d.name.startswith(prefix + '_'):
                continue
            val_str = d.name[len(prefix) + 1:]
            try:
                val = float(val_str)
            except ValueError:
                continue
            for seed_dir in sorted(d.iterdir()):
                npz_path = seed_dir / 'test_attack_metrics.npz'
                gv_path = seed_dir / 'gradvis.npy'
                if not npz_path.exists() or not gv_path.exists():
                    continue
                metrics = np.load(npz_path)
                per_byte_acc = np.array([float(metrics[f'test/acc/{b}']) for b in range(NUM_BYTES)])
                per_byte_rank = np.array([float(metrics[f'test/rank/{b}']) for b in range(NUM_BYTES)])
                full_acc = float(metrics['test/acc'])
                mtd = float(metrics['test/mtd'])
                estimates = np.load(gv_path)
                per_byte_oa = oracle_metric(estimates)
                all_data[prefix].append((val, seed_dir.name, per_byte_acc, full_acc, per_byte_oa, mtd, per_byte_rank))
                print(f'  {d.name}/{seed_dir.name}: acc={full_acc:.4f}, mtd={mtd:.1f}, mean_oa={per_byte_oa.mean():.4f}')
    return all_data


def plot_sweep_overlay(all_data, defaults):
    """Plot regularization sweeps: classification accuracy and oracle agreement side by side."""
    fig, axes = plt.subplots(2, len(SWEEP_PREFIXES), figsize=(3.2 * len(SWEEP_PREFIXES), 6), squeeze=False)

    # Also compute baselines
    oracle_metric = OracleAgreement(SNR_DIR, DATASET_ID)
    random_oa = oracle_metric.get_random_oracle_agreement()
    profiling_oa = oracle_metric.get_profiling_oracle_agreement()

    for col, prefix in enumerate(SWEEP_PREFIXES):
        runs = all_data[prefix]
        if not runs:
            continue

        # Group by reg value
        by_val: Dict[float, List] = {}
        for val, seed, pba, fa, pboa, *_ in runs:
            by_val.setdefault(val, []).append((pba, fa, pboa))

        sorted_vals = sorted(by_val.keys())
        acc_means = np.array([np.mean([r[1] for r in by_val[v]]) for v in sorted_vals])
        acc_stds = np.array([np.std([r[1] for r in by_val[v]]) for v in sorted_vals])
        oa_means = np.array([np.mean([r[2].mean() for r in by_val[v]]) for v in sorted_vals])
        oa_stds = np.array([np.std([r[2].mean() for r in by_val[v]]) for v in sorted_vals])

        use_log = prefix in LOG_SCALE_PREFIXES
        plot_vals = sorted_vals

        # Top row: accuracy
        ax_acc = axes[0, col]
        ax_acc.plot(plot_vals, acc_means, 'o-', color='C0', markersize=4)
        ax_acc.fill_between(plot_vals, acc_means - acc_stds, acc_means + acc_stds, alpha=0.2, color='C0')
        default_val = defaults[prefix]
        if default_val in plot_vals:
            idx = plot_vals.index(default_val)
            ax_acc.plot(default_val, acc_means[idx], '*', color='red', markersize=10, zorder=5)
        if col == 0:
            ax_acc.set_ylabel('Full-key accuracy')
        ax_acc.set_title(prefix.replace('_', ' ').title(), fontsize=10)
        ax_acc.tick_params(axis='x', labelrotation=45, labelsize=7)
        ax_acc.grid(True, alpha=0.3)
        if use_log:
            positive = [(v, m, s) for v, m, s in zip(plot_vals, acc_means, acc_stds) if v > 0]
            if positive:
                ax_acc.clear()
                pv, pm, ps = zip(*positive)
                ax_acc.plot(pv, pm, 'o-', color='C0', markersize=4)
                ax_acc.fill_between(pv, np.array(pm) - np.array(ps), np.array(pm) + np.array(ps), alpha=0.2, color='C0')
                ax_acc.set_xscale('log')
                if col == 0:
                    ax_acc.set_ylabel('Full-key accuracy')
                ax_acc.set_title(prefix.replace('_', ' ').title(), fontsize=10)
                ax_acc.tick_params(axis='x', labelrotation=45, labelsize=7)
                ax_acc.grid(True, alpha=0.3)

        # Bottom row: oracle agreement
        ax_oa = axes[1, col]
        ax_oa.plot(plot_vals, oa_means, 'o-', color='C1', markersize=4)
        ax_oa.fill_between(plot_vals, oa_means - oa_stds, oa_means + oa_stds, alpha=0.2, color='C1')
        ax_oa.axhline(profiling_oa.mean(), color='green', ls='--', alpha=0.6, lw=1, label='Oracle')
        ax_oa.axhline(random_oa.mean(), color='red', ls='--', alpha=0.6, lw=1, label='Random')
        if col == 0:
            ax_oa.set_ylabel('Mean oracle agreement')
        ax_oa.set_xlabel(prefix.replace('_', ' ').title())
        ax_oa.tick_params(axis='x', labelrotation=45, labelsize=7)
        ax_oa.grid(True, alpha=0.3)
        if col == len(SWEEP_PREFIXES) - 1:
            ax_oa.legend(fontsize=7)
        if use_log:
            positive = [(v, m, s) for v, m, s in zip(plot_vals, oa_means, oa_stds) if v > 0]
            if positive:
                ax_oa.clear()
                pv, pm, ps = zip(*positive)
                ax_oa.plot(pv, pm, 'o-', color='C1', markersize=4)
                ax_oa.fill_between(pv, np.array(pm) - np.array(ps), np.array(pm) + np.array(ps), alpha=0.2, color='C1')
                ax_oa.axhline(profiling_oa.mean(), color='green', ls='--', alpha=0.6, lw=1, label='Oracle')
                ax_oa.axhline(random_oa.mean(), color='red', ls='--', alpha=0.6, lw=1, label='Random')
                ax_oa.set_xscale('log')
                if col == 0:
                    ax_oa.set_ylabel('Mean oracle agreement')
                ax_oa.set_xlabel(prefix.replace('_', ' ').title())
                ax_oa.tick_params(axis='x', labelrotation=45, labelsize=7)
                ax_oa.grid(True, alpha=0.3)
                if col == len(SWEEP_PREFIXES) - 1:
                    ax_oa.legend(fontsize=7)

    fig.suptitle('Regularization Sweep: Classification vs Localization (ascadv1-fixed)', fontsize=12)
    fig.tight_layout()
    fig.savefig('experiments/_tmp_reg_sweep_overlay.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('Saved experiments/_tmp_reg_sweep_overlay.png')


def _collect_all_runs(all_data):
    """Flatten all runs across prefixes into arrays."""
    all_pba, all_pboa, all_fa, all_mtd, all_pbr = [], [], [], [], []
    for prefix, runs in all_data.items():
        for val, seed, pba, fa, pboa, mtd, pbr in runs:
            all_pba.append(pba)
            all_pboa.append(pboa)
            all_fa.append(fa)
            all_mtd.append(mtd)
            all_pbr.append(pbr)
    return (np.array(all_pba), np.array(all_pboa),
            np.array(all_fa), np.array(all_mtd), np.array(all_pbr))


def _make_scatter_grid(x_per_byte, y_per_byte, x_agg, y_agg,
                       x_label, y_label, x_agg_label, title, filename,
                       x_agg_log=False, per_byte_x_log=False):
    """Generic helper for per-byte + aggregate scatter grid."""
    from scipy.stats import spearmanr
    nrows, ncols = 4, 5
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.2 * nrows))
    axes_flat = axes.flatten()

    for b in range(NUM_BYTES):
        ax = axes_flat[b]
        x, y = x_per_byte[:, b], y_per_byte[:, b]
        ax.scatter(x, y, s=10, alpha=0.5, edgecolors='none')
        if len(x) > 3:
            rho, _ = spearmanr(x, y)
            ax.set_title(f'Byte {b} ($\\rho$={rho:.2f})', fontsize=9)
        else:
            ax.set_title(f'Byte {b}', fontsize=9)
        ax.set_xlabel(x_label, fontsize=8)
        ax.set_ylabel(y_label, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        if per_byte_x_log:
            ax.set_xscale('log')

    ax_agg = axes_flat[NUM_BYTES]
    ax_agg.scatter(x_agg, y_agg, s=10, alpha=0.5, edgecolors='none', color='C1')
    if len(x_agg) > 3:
        rho, _ = spearmanr(x_agg, y_agg)
        ax_agg.set_title(f'Aggregate ($\\rho$={rho:.2f})', fontsize=9)
    ax_agg.set_xlabel(x_agg_label, fontsize=8)
    ax_agg.set_ylabel('Mean oracle agreement', fontsize=8)
    ax_agg.tick_params(labelsize=7)
    ax_agg.grid(True, alpha=0.3)
    if x_agg_log:
        ax_agg.set_xscale('log')

    for i in range(NUM_BYTES + 1, nrows * ncols):
        axes_flat[i].set_visible(False)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {filename}')


def plot_scatter_per_byte(all_data):
    """Scatter plots: accuracy, MTD, and zoomed variants."""
    from scipy.stats import spearmanr
    pba, pboa, fa, mtd, pbr = _collect_all_runs(all_data)
    mean_oa = pboa.mean(axis=1)
    print(f'Total runs for scatter plots: {len(fa)}')

    # 1) Accuracy scatter (original)
    _make_scatter_grid(
        pba, pboa, fa, mean_oa,
        'Accuracy', 'Oracle agreement', 'Full-key accuracy',
        'Per-byte Accuracy vs Oracle Agreement (all runs)',
        'experiments/_tmp_reg_sweep_scatter.png')

    # 2) Accuracy scatter — zoomed to high-accuracy regime
    nrows, ncols = 4, 5
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.2 * nrows))
    axes_flat = axes.flatten()
    for b in range(NUM_BYTES):
        ax = axes_flat[b]
        mask = pba[:, b] > 0.8
        x, y = pba[mask, b], pboa[mask, b]
        ax.scatter(x, y, s=12, alpha=0.5, edgecolors='none')
        if mask.sum() > 3:
            rho, _ = spearmanr(x, y)
            ax.set_title(f'Byte {b} ($\\rho$={rho:.2f}, n={mask.sum()})', fontsize=9)
        else:
            ax.set_title(f'Byte {b} (n={mask.sum()})', fontsize=9)
        ax.set_xlabel('Accuracy', fontsize=8)
        ax.set_ylabel('Oracle agreement', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
    ax_agg = axes_flat[NUM_BYTES]
    mask_agg = fa > 0.5
    ax_agg.scatter(fa[mask_agg], mean_oa[mask_agg], s=12, alpha=0.5, edgecolors='none', color='C1')
    if mask_agg.sum() > 3:
        rho, _ = spearmanr(fa[mask_agg], mean_oa[mask_agg])
        ax_agg.set_title(f'Aggregate ($\\rho$={rho:.2f}, n={mask_agg.sum()})', fontsize=9)
    ax_agg.set_xlabel('Full-key accuracy', fontsize=8)
    ax_agg.set_ylabel('Mean oracle agreement', fontsize=8)
    ax_agg.tick_params(labelsize=7)
    ax_agg.grid(True, alpha=0.3)
    for i in range(NUM_BYTES + 1, nrows * ncols):
        axes_flat[i].set_visible(False)
    fig.suptitle('Accuracy vs Oracle Agreement — high-accuracy regime', fontsize=12)
    fig.tight_layout()
    fig.savefig('experiments/_tmp_reg_sweep_scatter_zoomed.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('Saved experiments/_tmp_reg_sweep_scatter_zoomed.png')

    # 3) MTD / Rank scatter (all runs)
    _make_scatter_grid(
        pbr, pboa, mtd, mean_oa,
        'Rank', 'Oracle agreement', 'MTD',
        'Per-byte Rank / MTD vs Oracle Agreement (all runs)',
        'experiments/_tmp_reg_sweep_scatter_mtd.png',
        x_agg_log=True, per_byte_x_log=True)

    # 4) MTD / Rank scatter — zoomed to non-failed models
    mask_mtd = mtd < 1000
    if mask_mtd.sum() > 5:
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.2 * nrows))
        axes_flat = axes.flatten()
        for b in range(NUM_BYTES):
            ax = axes_flat[b]
            bmask = pbr[:, b] < 50
            x, y = pbr[bmask, b], pboa[bmask, b]
            ax.scatter(x, y, s=12, alpha=0.5, edgecolors='none')
            if bmask.sum() > 3:
                rho, _ = spearmanr(x, y)
                ax.set_title(f'Byte {b} ($\\rho$={rho:.2f}, n={bmask.sum()})', fontsize=9)
            else:
                ax.set_title(f'Byte {b} (n={bmask.sum()})', fontsize=9)
            ax.set_xlabel('Rank', fontsize=8)
            ax.set_ylabel('Oracle agreement', fontsize=8)
            ax.set_xscale('log')
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
        ax_agg = axes_flat[NUM_BYTES]
        ax_agg.scatter(mtd[mask_mtd], mean_oa[mask_mtd], s=12, alpha=0.5, edgecolors='none', color='C1')
        rho, _ = spearmanr(mtd[mask_mtd], mean_oa[mask_mtd])
        ax_agg.set_title(f'Aggregate ($\\rho$={rho:.2f}, n={mask_mtd.sum()})', fontsize=9)
        ax_agg.set_xlabel('MTD', fontsize=8)
        ax_agg.set_ylabel('Mean oracle agreement', fontsize=8)
        ax_agg.set_xscale('log')
        ax_agg.tick_params(labelsize=7)
        ax_agg.grid(True, alpha=0.3)
        for i in range(NUM_BYTES + 1, nrows * ncols):
            axes_flat[i].set_visible(False)
        fig.suptitle('Rank / MTD vs Oracle Agreement — non-failed models', fontsize=12)
        fig.tight_layout()
        fig.savefig('experiments/_tmp_reg_sweep_scatter_mtd_zoomed.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
        print('Saved experiments/_tmp_reg_sweep_scatter_mtd_zoomed.png')


def plot_per_byte_breakdown(all_data):
    """Per-byte heatmaps: for each regularization type, show per-byte accuracy and oracle agreement."""
    oracle_metric = OracleAgreement(SNR_DIR, DATASET_ID)
    profiling_oa = oracle_metric.get_profiling_oracle_agreement()

    for prefix in SWEEP_PREFIXES:
        runs = all_data[prefix]
        if not runs:
            continue

        by_val: Dict[float, List] = {}
        for val, seed, pba, fa, pboa, *_ in runs:
            by_val.setdefault(val, []).append((pba, fa, pboa))

        sorted_vals = sorted(by_val.keys())
        n_vals = len(sorted_vals)

        # Average over seeds for each reg value
        acc_grid = np.zeros((n_vals, NUM_BYTES))
        oa_grid = np.zeros((n_vals, NUM_BYTES))
        for i, v in enumerate(sorted_vals):
            acc_grid[i] = np.mean([r[0] for r in by_val[v]], axis=0)
            oa_grid[i] = np.mean([r[2] for r in by_val[v]], axis=0)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(3, 0.5 * n_vals + 1.5)))

        val_labels = [f'{v:g}' for v in sorted_vals]

        im1 = ax1.imshow(acc_grid, aspect='auto', cmap='viridis')
        ax1.set_xticks(range(NUM_BYTES))
        ax1.set_xticklabels(range(NUM_BYTES), fontsize=8)
        ax1.set_yticks(range(n_vals))
        ax1.set_yticklabels(val_labels, fontsize=8)
        ax1.set_xlabel('Byte')
        ax1.set_ylabel('Reg. strength')
        ax1.set_title('Per-byte accuracy')
        plt.colorbar(im1, ax=ax1, shrink=0.8)

        im2 = ax2.imshow(oa_grid, aspect='auto', cmap='viridis')
        ax2.set_xticks(range(NUM_BYTES))
        ax2.set_xticklabels(range(NUM_BYTES), fontsize=8)
        ax2.set_yticks(range(n_vals))
        ax2.set_yticklabels(val_labels, fontsize=8)
        ax2.set_xlabel('Byte')
        ax2.set_ylabel('Reg. strength')
        ax2.set_title('Per-byte oracle agreement')
        plt.colorbar(im2, ax=ax2, shrink=0.8)

        fig.suptitle(f'{prefix.replace("_", " ").title()} — Per-byte Breakdown', fontsize=12)
        fig.tight_layout()
        fig.savefig(f'experiments/_tmp_reg_sweep_perbyte_{prefix}.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved experiments/_tmp_reg_sweep_perbyte_{prefix}.png')


if __name__ == '__main__':
    defaults = load_defaults()
    print('Loading all runs and computing oracle agreement...')
    all_data = load_all_runs()
    print('\nPlotting sweep overlays...')
    plot_sweep_overlay(all_data, defaults)
    print('\nPlotting scatter plots...')
    plot_scatter_per_byte(all_data)
    print('\nPlotting per-byte breakdowns...')
    plot_per_byte_breakdown(all_data)
    print('\nDone! Files to clean up:')
    print('  experiments/_tmp_reg_sweep_*.png')
    print('  experiments/_tmp_reg_sweep_analysis.py')
    print('  experiments/_tmp_plot_oracle_agreement.py')
    print('  experiments/_tmp_oracle_agreement_plots.png')