import argparse
from pathlib import Path
from typing import Optional, Dict, Any

import yaml
from lightning import Trainer

from experiments.initialization import *
from .supervised import construct_datasets, construct_loaders
from leakage_localization.training.supervised_lightning_module import SupervisedModule

def load_training_module(
        ckpt_path: Path,
) -> SupervisedModule:
    module = SupervisedModule.load_from_checkpoint(ckpt_path, map_location='cpu', weights_only=False)
    return module

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', required=True, type=Path)
    parser.add_argument('--dest', type=Path)
    parser.add_argument('--config-file', required=True, type=str)
    parser.add_argument('--config-root', type=Path, default=LOCAL_CONFIG_ROOT)
    append_directory_clargs(parser)
    args = parser.parse_args()

    ckpt_path: Path = args.ckpt_path
    assert ckpt_path.exists()
    dest: Optional[Path] = args.dest
    if dest is None:
        dest = ckpt_path.parent
    dest.mkdir(exist_ok=True, parents=True)
    config_file: str = args.config_file
    config_root: Path = args.config_root
    config_path = config_root / f'{config_file}.yaml'
    assert config_path.exists()

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    module = load_training_module(ckpt_path)
    train_set, val_set, test_set = construct_datasets(config)
    train_loader, val_loader, test_loader = construct_loaders(train_set, val_set, test_set, config)
    trainer = Trainer(
        accelerator='gpu',
        precision='bf16-mixed',
        default_root_dir=dest,
        logger=False
    )
    trainer.test(module, dataloaders=test_loader)

if __name__ == '__main__':
    main()