import argparse
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import yaml
from numpy.typing import NDArray
from lightning import Trainer
from matplotlib import pyplot as plt
import pandas
from torch.utils.data import Dataset, DataLoader

from experiments.initialization import *
from experiments.initialization.directories import safe_load_yaml
from .supervised import construct_datasets, construct_loaders
from leakage_localization.training.supervised_lightning_module import SupervisedModule
from leakage_localization.deep_attribution.attributor import Attributor

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
        trace_statistics=trace_statistics
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

def compute_feature_attribution(
        module: SupervisedModule,
        profiling_dataloader: DataLoader,
        dest_dir: Path
):
    rv = dict()
    for method in ['gradvis']:
        dest = dest_dir / f'{method}.npy'
        if dest.exists():
            continue
        module.to('cuda')
        attributor = Attributor(module)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            attr = attributor(method, profiling_dataloader, show_progress_bar=True).float().cpu().numpy()
        np.save(dest, attr)

def test_model(
        module: SupervisedModule,
        loaders: Dict[str, DataLoader],
        dest_dir: Path
):
    for loader_key, loader in loaders.items():
        module.to('cuda')
        trainer = Trainer(
            accelerator='gpu',
            precision='bf16-mixed',
            default_root_dir=dest_dir,
            logger=False
        )
        results = trainer.test(module, dataloaders=loader)
        metrics = {k: np.array(v) for k, v in results[0].items()}
        np.savez(dest_dir / f'{loader_key}_attack_metrics.npz', **metrics)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', required=True, type=Path)
    parser.add_argument('--dest', type=Path)
    parser.add_argument('--do-attack-evaluation', default=False, action='store_true')
    parser.add_argument('--do-feature-attribution', default=False, action='store_true')
    append_directory_clargs(parser)
    args = parser.parse_args()

    ckpt_path: Path = args.ckpt_path
    assert ckpt_path.exists()
    dest: Optional[Path] = args.dest
    do_attack_evaluation: bool = args.do_attack_evaluation
    do_feature_attribution: bool = args.do_feature_attribution
    if dest is None:
        dest = ckpt_path.parent
    dest.mkdir(exist_ok=True, parents=True)
    config_path = ckpt_path.parent / 'config.yaml'
    assert config_path.exists()

    with open(config_path, 'r') as f:
        config = safe_load_yaml(f)
    config['training']['batch_size'] = 128
    
    profiling_set, attack_set, train_set, val_set, test_set, trace_statistics = construct_datasets(config, val_partition='attack')
    train_loader, val_loader, test_loader, profiling_loader = construct_loaders([], [train_set, val_set, test_set, profiling_set], config)
    module = load_training_module(ckpt_path, profiling_set=profiling_set, trace_statistics=trace_statistics, config=config)
    if do_attack_evaluation:
        test_model(module, {'train': train_loader, 'val': val_loader, 'test': test_loader}, dest)
    if do_feature_attribution:
        compute_feature_attribution(module, profiling_loader, dest)

    metrics_path = ckpt_path.parent / 'metrics.csv'
    if metrics_path.exists():
        plot_training_curves(metrics_path, dest)

if __name__ == '__main__':
    main()