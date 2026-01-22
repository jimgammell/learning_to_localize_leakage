from pathlib import Path
import argparse
from typing import Optional, Dict

import yaml

PROJ_ROOT = Path(__file__).resolve().parent.parent
LOCAL_CONFIG_ROOT = PROJ_ROOT / 'local_config'
ASCADV1_FIXED_CROPPED_ROOT: Optional[Path] = None
ASCADV1_FIXED_UNCROPPED_ROOT: Optional[Path] = None
ASCADV1_VARIABLE_CROPPED_ROOT: Optional[Path] = None
ASCADV1_VARIABLE_UNCROPPED_ROOT: Optional[Path] = None
DOWNLOADS_CACHE_ROOT: Optional[Path] = None
OUTPUTS_ROOT: Optional[Path] = None

def append_directory_clargs(p: argparse.ArgumentParser):
    for key, description in zip(
        ('ascadv1-fixed-cropped-root', 'Root directory for the cropped version of ASCADv1 (fixed key).'),
        ('ascadv1-fixed-uncropped-root', 'Root directory for the uncropped version of ASCADv1 (fixed key).'),
        ('ascadv1-variable-cropped-root', 'Root directory for the cropped version of ASCADv1 (variable key).'),
        ('ascadv1-variable-uncropped-root', 'Root directory for the uncropped version of ASCADv1 (variable key).'),
        ('downloads-cache-root', 'Root directory for caching downloaded files.'),
        ('outputs-root', 'Root directory for experiment outputs.')
    ):
        p.add_argument(
            f'--{key}',
            help=description,
            action='store',
            type=Optional[str],
            default=None
        )

def load_directory_config() -> Dict[str, Path]:
    config_path = LOCAL_CONFIG_ROOT / 'directories.yaml'
    if not config_path.exists():
        return dict()
    with open(config_path) as config_file:
        directories = yaml.safe_load(config_file)
        return {k: Path(v) for k, v in directories.items()}