import argparse
from pathlib import Path
from typing import Optional, Dict, Any

import yaml
from numpy.typing import NDArray
from lightning import Trainer
from matplotlib import pyplot as plt
import pandas
from torch.utils.data import Dataset

from experiments.initialization import *
from experiments.initialization.directories import safe_load_yaml
from .supervised import construct_datasets, construct_loaders
from leakage_localization.training.supervised_lightning_module import SupervisedModule

def load_training_module(
        ckpt_path: Path,
        profiling_set: Dataset,
        trace_statistics: Dict[str, NDArray[np.floating]],
        config: Dict[str, Any]
) -> SupervisedModule:
    module = SupervisedModule.load_from_checkpoint(
        ckpt_path,
        map_location='cpu',
        weights_only=False,
        trace_statistics=trace_statistics,
        mtd_kwargs={
            'target_preds_to_key_preds': profiling_set.target_preds_to_key_preds,
            'int_var_keys': profiling_set.int_var_keys,
            'attack_count': config['mtd']['attack_count'],
            'traces_per_attack': config['mtd']['traces_per_attack']
        },
        mixup_alpha=config['training'].get('mixup_alpha', 0.0),
        random_roll_scale=float(config['data'].get('random_roll_scale', 0.0)),
        random_lpf_scale=float(config['data'].get('random_lpf_scale', 0.0)),
        compute_val_mtd=config['training'].get('early_stop_metric', 'val/rank') == 'val/mtd',
    )
    return module

def plot_training_curves(
        metrics_path: Path,
        dest: Path
):
    metrics_df = pandas.read_csv(metrics_path)
    train_df = metrics_df.dropna(subset=['train/loss'])
    val_df = metrics_df.dropna(subset=['val/loss'])

    for metric in ['loss', 'acc', 'rank']:
        fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 0.6*WIDTH))
        axes[0].plot(train_df['epoch'], train_df[f'train/{metric}'], label='train', color='blue', linestyle=':')
        axes[0].plot(val_df['epoch'], val_df[f'val/{metric}'], label='val', color='blue', linestyle='-')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel(f'Metric: {metric}')
        axes[0].set_title('Avg. over bytes')
        axes[0].legend()

        for byte in range(16):
            axes[1].plot(train_df['epoch'], train_df[f'train/{metric}/{byte}'], color='blue', linestyle=':')
            axes[1].plot(val_df['epoch'], val_df[f'val/{metric}/{byte}'], color='blue', linestyle='-')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel(f'Metric: {metric}')
        axes[1].set_title('Per-byte')

        fig.tight_layout()
        fig.savefig(dest / f'{metric}_curves.pdf', dpi=300)
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', required=True, type=Path)
    parser.add_argument('--dest', type=Path)
    append_directory_clargs(parser)
    args = parser.parse_args()

    ckpt_path: Path = args.ckpt_path
    assert ckpt_path.exists()
    dest: Optional[Path] = args.dest
    if dest is None:
        dest = ckpt_path.parent
    dest.mkdir(exist_ok=True, parents=True)
    config_path = ckpt_path.parent / 'config.yaml'
    assert config_path.exists()

    with open(config_path, 'r') as f:
        config = safe_load_yaml(f)
    
    profiling_set, attack_set, train_set, val_set, test_set, trace_statistics = construct_datasets(config, val_partition='profile')
    train_loader, val_loader, test_loader = construct_loaders(train_set, val_set, test_set, config)
    module = load_training_module(ckpt_path, profiling_set=profiling_set, trace_statistics=trace_statistics, config=config)
    trainer = Trainer(
        accelerator='gpu',
        precision='bf16-mixed',
        default_root_dir=dest,
        logger=False
    )
    trainer.test(module, dataloaders=test_loader)

    metrics_path = ckpt_path.parent / 'metrics.csv'
    if metrics_path.exists():
        plot_training_curves(metrics_path, dest)

if __name__ == '__main__':
    main()