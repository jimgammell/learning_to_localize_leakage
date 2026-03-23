from pathlib import Path
from typing import Tuple

from matplotlib import pyplot as plt
import numpy as np
from numpy.typing import NDArray

from experiments.initialization import *

SWEEP_PREFIXES = [
    'input_dropout',
    'hidden_dropout',
    'weight_decay',
    'gaussian_noise',
    'random_roll',
    'random_lpf',
    'mixup',
]

DATASETS = [
    ('ascadv1_fixed',    'ASCADv1-fixed',    'test/acc', 'Accuracy'),
    ('ascadv1_variable', 'ASCADv1-variable', 'test/acc', 'Accuracy'),
    ('ches_ctf_2018',    'CHES-CTF-2018',    'test/mtd', 'MTD'),
]

def load_sweep(base_dir: Path, prefix: str, metric_key: str) -> Tuple[NDArray, NDArray, NDArray]:
    """Discover dirs matching prefix, parse values, load metrics. Returns (values, means, stds) sorted by value."""
    if not base_dir.exists():
        return np.array([]), np.array([]), np.array([])
    entries = []
    for d in base_dir.iterdir():
        if not d.is_dir() or not d.name.startswith(prefix + '_'):
            continue
        val_str = d.name[len(prefix) + 1:]
        try:
            val = float(val_str)
        except ValueError:
            continue
        seed_metrics = []
        for seed_dir in sorted(d.iterdir()):
            npz_path = seed_dir / 'test_attack_metrics.npz'
            if not npz_path.exists():
                continue
            data = np.load(npz_path)
            if metric_key in data:
                seed_metrics.append(float(data[metric_key]))
        if seed_metrics:
            entries.append((val, np.mean(seed_metrics), np.std(seed_metrics)))
    if not entries:
        return np.array([]), np.array([]), np.array([])
    entries.sort(key=lambda x: x[0])
    vals, means, stds = zip(*entries)
    return np.array(vals), np.array(means), np.array(stds)


def plot_sweeps():
    fig, axes = plt.subplots(len(DATASETS), len(SWEEP_PREFIXES), figsize=(WIDTH * 2.5, 0.8 * WIDTH * len(DATASETS) / len(SWEEP_PREFIXES) * 2.5), squeeze=False)

    for row, (dataset_key, dataset_label, metric_key, metric_label) in enumerate(DATASETS):
        base_dir = Path(f'./outputs/{dataset_key}/reg_sweep')
        for col, prefix in enumerate(SWEEP_PREFIXES):
            ax = axes[row, col]
            vals, means, stds = load_sweep(base_dir, prefix, metric_key)
            if len(vals) == 0:
                ax.set_visible(False)
                continue
            ax.plot(vals, means, color='blue', marker='o', markersize=3)
            ax.fill_between(vals, means - stds, means + stds, alpha=0.2, color='blue')
            if row == len(DATASETS) - 1:
                ax.set_xlabel(prefix.replace('_', ' ').title())
            if col == 0:
                ax.set_ylabel(f'{dataset_label}\n{metric_label}')
            if any(v > 0 and v < 0.01 for v in vals):
                ax.set_xscale('symlog', linthresh=min(v for v in vals if v > 0))
            if metric_key == 'test/mtd':
                ax.set_yscale('log')
            ax.tick_params(axis='x', labelrotation=45, labelsize=7)

    fig.tight_layout()
    fig.savefig('./outputs/reg_sweep_summary.pdf', dpi=DPI)
    plt.close(fig)
    print('Saved to ./outputs/reg_sweep_summary.pdf')


if __name__ == '__main__':
    plot_sweeps()