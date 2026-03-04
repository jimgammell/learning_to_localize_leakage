import argparse
from pathlib import Path
from typing import Callable, Dict, Any, Optional

import yaml

from experiments.initialization import *
from experiments.initialization.directories import safe_load_yaml

def _apply_overrides(config: Dict[str, Any], overrides: list) -> None:
    i = 0
    while i < len(overrides):
        arg = overrides[i]
        if not arg.startswith('--'):
            raise ValueError(f'Expected --key.subkey value, got: {arg}')
        key = arg[2:]
        if i + 1 >= len(overrides) or overrides[i + 1].startswith('--'):
            raise ValueError(f'No value provided for override {arg}')
        raw_value = overrides[i + 1]
        i += 2
        keys = key.split('.')
        d = config
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                raise ValueError(f'Invalid config path: {key}')
            d = d[k]
        if keys[-1] not in d:
            raise ValueError(f'Key {key} not found in config')
        d[keys[-1]] = yaml.safe_load(raw_value)


def main(
        run_fn: Callable[[Path, Dict[str, Any]], None]
):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dest', required=True, type=Path)
    parser.add_argument('--optuna-study-path', type=Path, default=None)
    parser.add_argument('--optuna-run-count', type=int, default=1)
    parser.add_argument('--enable-pruning', action='store_true', default=False)
    parser.add_argument('--config-file', required=True, type=str)
    parser.add_argument('--config-root', type=Path, default=LOCAL_CONFIG_ROOT)
    append_directory_clargs(parser)
    args, overrides = parser.parse_known_args()

    dest: Path = args.dest
    dest.mkdir(exist_ok=True, parents=True)
    optuna_study_path: Optional[Path] = args.optuna_study_path
    optuna_run_count: int = args.optuna_run_count
    enable_pruning: bool = args.enable_pruning
    assert optuna_run_count > 0
    config_root: Path = args.config_root
    config_path = config_root / f'{args.config_file}.yaml'
    assert config_path.exists()

    with open(config_path, 'r') as f:
        config = safe_load_yaml(f)

    _apply_overrides(config, overrides)

    if optuna_study_path is not None:
        assert len(overrides) == 0

    run_fn(
        dest=dest,
        config=config,
        optuna_study_path=optuna_study_path,
        optuna_run_count=optuna_run_count,
        enable_pruning=enable_pruning
    )