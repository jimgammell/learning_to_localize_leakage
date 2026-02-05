import argparse
from pathlib import Path
from typing import Callable, Dict, Any

import yaml

from experiments.initialization import *

def main(
        run_fn: Callable[[Path, Dict[str, Any]], None]
):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dest', required=True, type=Path)
    parser.add_argument('--config-file', required=True, type=str)
    parser.add_argument('--config-root', type=Path, default=LOCAL_CONFIG_ROOT)
    append_directory_clargs(parser)
    args = parser.parse_args()

    dest: Path = args.dest
    dest.mkdir(exist_ok=True, parents=True)
    config_root: Path = args.config_root
    config_path = config_root / f'{args.config_file}.yaml'
    assert config_path.exists()

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    run_fn(
        dest=dest,
        config=config
    )