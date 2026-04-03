"""Generate per-plot PDF files for a trained-model run directory.

One PDF is produced per plot type.  Missing files are silently skipped.
Baseline results (random / oracle) are overlaid as dashed reference lines
on every comparative plot when a baseline directory can be found.

Usage:
    python experiments/visualize_trained_model.py \
        --run-dir ./outputs/ascadv1_fixed/strong_attacker/seed_0

    # Override auto-detected directories:
    python experiments/visualize_trained_model.py \
        --run-dir ./outputs/ascadv1_fixed/strong_attacker/seed_0 \
        --snr-dir ./outputs/ascadv1_fixed/snr \
        --baseline-dir ./outputs/ascadv1_fixed/baselines
"""
import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas
import yaml
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.visualize_runs import (
    plot_leakiness_over_time,
    plot_wb_scatterplots,
    plot_wb_comparison_grid,
    plot_rank_trajectories,
    plot_per_byte_bar,
    plot_occlusion_test,
)

# ── style constants ───────────────────────────────────────────────────────────

# Colors for attribution methods (cycle if more than 4)
METHOD_COLORS = ['royalblue', 'darkorange', 'crimson', 'mediumpurple']
BASELINE_STYLES = {
    'random': dict(color='0.55', linestyle='--', linewidth=1.1, alpha=0.85),
    'oracle': dict(color='forestgreen', linestyle='--', linewidth=1.1, alpha=0.85),
}

# ── helpers ───────────────────────────────────────────────────────────────────

def _save(fig, dest: Path, name: str):
    path = dest / f'{name}.pdf'
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


def _detect_attr_methods(run_dir: Path) -> List[str]:
    return sorted(f.stem for f in run_dir.iterdir()
                  if f.suffix == '.npy' and '.' not in f.stem)


def _load_dataset_id(run_dir: Path) -> Optional[str]:
    cfg = run_dir / 'config.yaml'
    if not cfg.exists():
        return None
    with open(cfg) as f:
        kw = yaml.safe_load(f)
    return kw.get('data', {}).get('id', None)


def _find_dir(run_dir: Path, arg: Optional[Path], *subpaths: str) -> Optional[Path]:
    """Return arg if provided and exists, else search ancestor dirs for subpaths."""
    if arg is not None:
        return arg if arg.exists() else None
    for candidate in [run_dir.parent / p for p in subpaths] + \
                     [run_dir.parent.parent / p for p in subpaths]:
        if candidate.exists():
            return candidate
    return None


def _load_oracle_leakiness(snr_dir: Path, dataset_id: str) -> Optional[np.ndarray]:
    try:
        from leakage_localization.evaluation.oracle_agreement import OracleAgreement
        return OracleAgreement(snr_dir, dataset_id).get_oracle_leakiness('attack')
    except Exception as e:
        print(f'  [skip wb-oracle] {e}')
        return None


def _load_npy(path: Path) -> Optional[np.ndarray]:
    return np.load(path) if path.exists() else None


def _load_npz(path: Path):
    return np.load(path, allow_pickle=True) if path.exists() else None


# ── per-plot generators ───────────────────────────────────────────────────────

def gen_training_curves(run_dir: Path, dest: Path):
    csv = run_dir / 'metrics.csv'
    if not csv.exists():
        return
    m = pandas.read_csv(csv)
    train_mask = ~m['train/loss'].isna()
    val_mask   = ~m['val/loss'].isna()
    t_step = m['step'][train_mask].values
    v_step = m['step'][val_mask].values

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))

    ax = axes[0]
    ax.plot(t_step, m['train/loss'][train_mask].values, color='royalblue',
            linestyle=':', linewidth=0.8, label='train', rasterized=True)
    ax.plot(v_step, m['val/loss'][val_mask].values, color='royalblue',
            linestyle='-', linewidth=1.2, label='val', rasterized=True)
    ax.set_xlabel('Step'); ax.set_ylabel('Loss'); ax.set_title('Loss')
    ax.legend(fontsize=7)

    ax = axes[1]
    byte_cols = [c for c in m.columns if re.fullmatch(r'val/rank/\d+', c)]
    for col in byte_cols:
        ax.plot(v_step, m[col][val_mask].values, color='royalblue',
                linewidth=0.2, alpha=0.4, rasterized=True)
    for col_suffix, ls, lw, label in [
        ('val/rank_min', '-',  1.0, 'best byte'),
        ('val/rank_med', '--', 1.0, 'median byte'),
        ('val/rank_max', ':',  1.0, 'worst byte'),
    ]:
        if col_suffix in m.columns:
            ax.plot(v_step, m[col_suffix][val_mask].values, color='royalblue',
                    linestyle=ls, linewidth=lw, label=label, rasterized=True)
    ax.set_xlabel('Step'); ax.set_ylabel('Rank'); ax.set_title('Val Rank')
    ax.legend(fontsize=7)

    ax = axes[2]
    ax.plot(t_step, m['train/acc'][train_mask].values, color='royalblue',
            linestyle=':', linewidth=0.8, label='train', rasterized=True)
    ax.plot(v_step, m['val/acc'][val_mask].values, color='royalblue',
            linestyle='-', linewidth=1.2, label='val', rasterized=True)
    ax.set_xlabel('Step'); ax.set_ylabel('Accuracy'); ax.set_title('Accuracy')
    ax.legend(fontsize=7)

    fig.tight_layout()
    _save(fig, dest, 'training_curves')


def gen_attack_performance(run_dir: Path, dest: Path):
    npz = run_dir / 'attack_metrics.npz'
    if not npz.exists():
        return
    d = np.load(npz, allow_pickle=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    plot_rank_trajectories(d['rank_over_time'], axes[0], label='worst-case byte')
    axes[0].set_title('Attack rank over traces')
    axes[0].legend(fontsize=7)
    plot_per_byte_bar(d['per_byte_mtd'], axes[1])
    axes[1].set_ylabel('MTD (traces)')
    axes[1].set_title('Per-byte MTD')
    fig.tight_layout()
    _save(fig, dest, 'attack_performance')


def gen_leakiness(run_dir: Path, dest: Path, method: str):
    attr_path = run_dir / f'{method}.npy'
    if not attr_path.exists():
        return
    attr = np.load(attr_path)
    fig, ax = plt.subplots(figsize=(14, 3))
    plot_leakiness_over_time(attr, ax, title=method.replace('_', ' '))
    ax.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, dest, f'leakiness_{method}')


def gen_wb_scatterplots(run_dir: Path, dest: Path, method: str, oracle: np.ndarray):
    attr_path = run_dir / f'{method}.npy'
    if not attr_path.exists():
        return
    attr = np.load(attr_path)
    byte_count = attr.shape[0]
    nrows, ncols = 4, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    plot_wb_scatterplots(attr, oracle, axes)
    for idx in range(byte_count, nrows * ncols):
        axes.flatten()[idx].set_visible(False)
    fig.suptitle(f'Estimated vs. oracle — {method.replace("_", " ")}', y=1.01)
    fig.tight_layout()
    _save(fig, dest, f'wb_scatter_{method}')


def gen_wb_comparison_grid(run_dir: Path, dest: Path, method: str, oracle: np.ndarray):
    attr_path = run_dir / f'{method}.npy'
    if not attr_path.exists():
        return
    attr = np.load(attr_path)
    byte_count = attr.shape[0]
    fig, axes = plt.subplots(2, byte_count, figsize=(byte_count * 2, 4))
    plot_wb_comparison_grid(attr, oracle, axes)
    fig.suptitle(f'Per-byte leakiness — {method.replace("_", " ")} (top) vs. oracle (bottom)', y=1.01)
    fig.tight_layout()
    _save(fig, dest, f'wb_comparison_{method}')


def gen_oracle_agreement(
        run_dir: Path,
        dest: Path,
        methods: List[str],
        baseline_dir: Optional[Path],
):
    """One figure per metric (Spearman / AUROC).
    Each figure shows one line per attribution method plus dashed baseline lines.
    """
    # Build ordered dict: label → npz data.  Baselines go first (drawn behind).
    sources: Dict[str, object] = {}
    if baseline_dir is not None:
        for name in ('random', 'oracle'):
            p = baseline_dir / f'white_box_agreement.{name}.npz'
            if p.exists():
                sources[name] = np.load(p)
    for method in methods:
        p = run_dir / f'white_box_agreement.{method}.npz'
        if p.exists():
            sources[method] = np.load(p)

    if not sources:
        return

    byte_x = np.arange(16)

    for metric_key, metric_label, ylim, ref_y in [
        ('spearman', 'Spearman ρ', (-1, 1), 0.0),
        ('auroc',    'AUROC',      (0, 1),  0.5),
    ]:
        fig, ax = plt.subplots(figsize=(11, 3.5))
        method_idx = 0
        for label, d in sources.items():
            if metric_key not in d:
                continue
            vals = d[metric_key]
            if label in BASELINE_STYLES:
                style = BASELINE_STYLES[label]
                ax.plot(byte_x, vals, marker='o', markersize=4,
                        label=label, **style)
            else:
                color = METHOD_COLORS[method_idx % len(METHOD_COLORS)]
                method_idx += 1
                ax.plot(byte_x, vals, color=color, linestyle='-', linewidth=1.8,
                        marker='o', markersize=5, label=label.replace('_', ' '))

        ax.axhline(ref_y, color='black', linewidth=0.5, linestyle=':')
        ax.set_xlim(-0.5, 15.5)
        ax.set_ylim(*ylim)
        ax.set_xticks(byte_x)
        ax.set_xlabel('Byte index')
        ax.set_ylabel(metric_label)

        title = f'Oracle agreement — {metric_label}'
        if metric_key == 'auroc':
            for d in sources.values():
                if hasattr(d, '__contains__') and 'auroc_percentile' in d:
                    title += f'  (binary labels p={float(d["auroc_percentile"])})'
                    break
        ax.set_title(title)
        ax.legend(fontsize=8)
        fig.tight_layout()
        _save(fig, dest, f'oracle_agreement_{metric_key}')


def gen_dnno_occl(
        run_dir: Path,
        dest: Path,
        method: str,
        baseline_dir: Optional[Path],
):
    """Two subplots (forward | reverse). Baselines as dashed overlays."""
    directions = {'fwd': 'forward', 'rev': 'reverse'}
    # Check if at least one direction exists for the main method
    if not any((run_dir / f'{pfx}_dnno_occl.{method}.npy').exists() for pfx in directions):
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    kw_solid = dict(linewidth=1.2, marker='.', markersize=4, rasterized=True)

    for ax, (pfx, dir_label) in zip(axes, directions.items()):
        # Baseline overlays (drawn first)
        if baseline_dir is not None:
            for bname, bstyle in BASELINE_STYLES.items():
                bp = baseline_dir / f'{pfx}_dnno_occl.{bname}.npy'
                if bp.exists():
                    arr = np.load(bp)
                    x = np.linspace(0, 1, len(arr) + 1)[1:]
                    ax.plot(x, arr, label=bname, rasterized=True, **bstyle)
        # Main method
        mp = run_dir / f'{pfx}_dnno_occl.{method}.npy'
        if mp.exists():
            arr = np.load(mp)
            x = np.linspace(0, 1, len(arr) + 1)[1:]
            ax.plot(x, arr, color='royalblue', label=method.replace('_', ' '), **kw_solid)
        ax.set_xlabel('Fraction of features occluded')
        ax.set_ylabel('MTD (traces)')
        ax.set_title(dir_label)
        ax.legend(fontsize=7)

    fig.suptitle(f'DNN occlusion — {method.replace("_", " ")}')
    fig.tight_layout()
    _save(fig, dest, f'dnno_occl_{method}')


def gen_ta_mtd(
        run_dir: Path,
        dest: Path,
        method: str,
        baseline_dir: Optional[Path],
):
    """Rank-over-time + per-byte MTD. Baselines as dashed overlays."""
    npz_path = run_dir / f'ta_mtd.{method}.npz'
    if not npz_path.exists():
        return

    d = np.load(npz_path, allow_pickle=True)
    # Support both new key scheme (ta-mtd/0 … ta-mtd/15) and old ('mtd' array)
    def _load_per_byte_mtd(npz):
        if 'ta-mtd/0' in npz:
            n = sum(1 for k in npz if k.startswith('ta-mtd/'))
            return np.array([float(npz[f'ta-mtd/{b}']) for b in range(n)])
        return npz['mtd']

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))

    # Left: rank-over-time
    ax = axes[0]
    if baseline_dir is not None:
        for bname, bstyle in BASELINE_STYLES.items():
            bd = _load_npz(baseline_dir / f'ta_mtd.{bname}.npz')
            if bd is not None:
                traces = np.arange(1, bd['rank_over_time'].shape[1] + 1)
                ax.plot(traces, bd['rank_over_time'].max(axis=0),
                        label=bname, rasterized=True, **bstyle)
    plot_rank_trajectories(d['rank_over_time'], ax,
                           color='royalblue', label=method.replace('_', ' '))
    ax.set_title(f'Template attack rank — {method.replace("_", " ")}')
    ax.legend(fontsize=7)

    # Right: per-byte MTD grouped bars (method + baselines)
    ax = axes[1]
    per_byte_mtd = _load_per_byte_mtd(d)
    byte_x = np.arange(len(per_byte_mtd))
    baselines_mtd = {}
    if baseline_dir is not None:
        for bname in BASELINE_STYLES:
            bd = _load_npz(baseline_dir / f'ta_mtd.{bname}.npz')
            if bd is not None:
                baselines_mtd[bname] = _load_per_byte_mtd(bd)

    n_groups = 1 + len(baselines_mtd)
    bar_w = 0.75 / n_groups
    for i, (bname, bstyle) in enumerate(BASELINE_STYLES.items()):
        if bname in baselines_mtd:
            offset = (i - n_groups / 2 + 0.5) * bar_w
            ax.bar(byte_x + offset, baselines_mtd[bname], width=bar_w,
                   color=bstyle['color'], alpha=0.7, label=bname)
    # Main method last (foreground)
    offset = ((n_groups - 1) - n_groups / 2 + 0.5) * bar_w
    ax.bar(byte_x + offset, per_byte_mtd, width=bar_w,
           color='royalblue', alpha=0.9, label=method.replace('_', ' '))
    ax.set_xticks(byte_x)
    ax.set_xlabel('Byte index')
    ax.set_ylabel('MTD (traces)')
    ax.set_title('Per-byte MTD')
    ax.legend(fontsize=7)

    fig.tight_layout()
    _save(fig, dest, f'ta_mtd_{method}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--dest', type=Path, default=None,
                        help='Output directory. Defaults to --run-dir.')
    parser.add_argument('--snr-dir', type=Path, default=None)
    parser.add_argument('--baseline-dir', type=Path, default=None,
                        help='Directory with random.npy / oracle.npy and their '
                             'evaluation results.  Auto-detected if omitted.')
    parser.add_argument('--methods', type=str, nargs='*', default=None)
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    assert run_dir.exists(), f'run-dir not found: {run_dir}'
    dest: Path = args.dest or run_dir
    dest.mkdir(parents=True, exist_ok=True)

    methods = args.methods if args.methods is not None else _detect_attr_methods(run_dir)
    print(f'Attribution methods: {methods}')

    baseline_dir = _find_dir(run_dir, args.baseline_dir, 'baselines')
    print(f'Baseline dir: {baseline_dir}')

    snr_dir    = _find_dir(run_dir, args.snr_dir, 'snr')
    dataset_id = _load_dataset_id(run_dir)
    oracle_leakiness: Optional[np.ndarray] = None
    if snr_dir is not None and dataset_id is not None:
        oracle_leakiness = _load_oracle_leakiness(snr_dir, dataset_id)
    else:
        print(f'  [skip wb-oracle] snr_dir={snr_dir}, dataset_id={dataset_id}')

    gen_training_curves(run_dir, dest)
    gen_attack_performance(run_dir, dest)
    gen_oracle_agreement(run_dir, dest, methods, baseline_dir)

    for method in methods:
        gen_leakiness(run_dir, dest, method)
        gen_dnno_occl(run_dir, dest, method, baseline_dir)
        gen_ta_mtd(run_dir, dest, method, baseline_dir)
        if oracle_leakiness is not None:
            gen_wb_scatterplots(run_dir, dest, method, oracle_leakiness)
            gen_wb_comparison_grid(run_dir, dest, method, oracle_leakiness)


if __name__ == '__main__':
    main()