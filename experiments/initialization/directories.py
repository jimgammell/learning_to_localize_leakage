from pathlib import Path
import argparse
from typing import Optional, Dict

import yaml

PROJ_ROOT = Path(__file__).resolve().parent.parent
LOCAL_CONFIG_ROOT = PROJ_ROOT / 'local_config'
LOCAL_DIRECTORY_CONFIG = LOCAL_CONFIG_ROOT / 'directories.yaml'
ASCADV1_FIXED_ROOT: Optional[Path] = None
ASCADV1_VARIABLE_ROOT: Optional[Path] = None
ASCADV2_ROOT: Optional[Path] = None
CHES_CTF_2018_ROOT: Optional[Path] = None
DPAV4d2_ROOT: Optional[Path] = None
DOWNLOADS_CACHE_ROOT: Optional[Path] = None
OUTPUTS_ROOT: Optional[Path] = None

def append_directory_clargs(p: argparse.ArgumentParser):
    for key, description in (
        ('ascadv1-fixed-root', 'Root directory for ASCADv1-fixed'),
        ('ascadv1-variable-root', 'Root directory for ASCADv1-variable'),
        ('ascadv2-root', 'Root directory for ASCADv2'),
        ('ches-ctf-2018-root', 'Root directory for CHES-CTF-2018'),
        ('dpav4_2-root', 'Root directory for DPAv4.2'),
        ('downloads-cache-root', 'Root directory for caching downloaded files'),
        ('outputs-root', 'Root directory for experiment outputs')
    ):
        p.add_argument(
            f'--{key}',
            help=description,
            action='store',
            type=str,
            default=None
        )

def load_directory_config() -> Dict[str, Path]:
    if not LOCAL_DIRECTORY_CONFIG.exists():
        return dict()
    with open(LOCAL_DIRECTORY_CONFIG) as config_file:
        directories = yaml.safe_load(config_file)
        return directories

def init_directories(clargs: Optional[Dict[str, str]] = None, config: Optional[Dict[str, str]] = None):
    global ASCADV1_FIXED_ROOT, ASCADV1_VARIABLE_ROOT, CHES_CTF_2018_ROOT, DPAV4d2_ROOT, DOWNLOADS_CACHE_ROOT, OUTPUTS_ROOT
    assert (clargs is not None) or (config is not None)
    if clargs is None:
        clargs = dict()
    if config is None:
        config = dict()
    for dirkey, dirpath in config.items():
        if dirkey == 'ascadv1-fixed-root':
            ASCADV1_FIXED_ROOT = Path(dirpath)
            ASCADV1_FIXED_ROOT.mkdir(exist_ok=True)
        elif dirkey == 'ascadv1-variable-root':
            ASCADV1_VARIABLE_ROOT = Path(dirpath)
            ASCADV1_VARIABLE_ROOT.mkdir(exist_ok=True)
        elif dirkey == 'ascadv2-root':
            ASCADV2_ROOT = Path(dirpath)
            ASCADV2_ROOT.mkdir(exist_ok=True)
        elif dirkey == 'ches-ctf-2018-root':
            CHES_CTF_2018_ROOT = Path(dirpath)
            CHES_CTF_2018_ROOT.mkdir(exist_ok=True)
        elif dirkey == 'dpav4_2-root':
            DPAV4d2_ROOT = Path(dirpath)
            DPAV4d2_ROOT.mkdir(exist_ok=True)
        elif dirkey == 'downloads-cache-root':
            DOWNLOADS_CACHE_ROOT = Path(dirpath)
            DOWNLOADS_CACHE_ROOT.mkdir(exist_ok=True)
        elif dirkey == 'outputs-root':
            OUTPUTS_ROOT = Path(dirpath)
            OUTPUTS_ROOT.mkdir(exist_ok=True)
        else:
            raise RuntimeError(f'Unrecognized directory configured in {LOCAL_DIRECTORY_CONFIG}: {dirkey}')
    for dirkey, dirpath in clargs.items():
        if dirkey == 'ascadv1-fixed-root':
            ASCADV1_FIXED_ROOT = Path(dirpath)
        elif dirkey == 'ascadv1-variable-root':
            ASCADV1_VARIABLE_ROOT = Path(dirpath)
        elif dirkey == 'ascadv2-root':
            ASCADV2_ROOT = Path(dirpath)
        elif dirkey == 'ches-ctf-2018-root':
            CHES_CTF_2018_ROOT = Path(dirpath)
        elif dirkey == 'dpav4_2-root':
            DPAV4d2_ROOT = Path(dirpath)
        elif dirkey == 'downloads-cache-root':
            DOWNLOADS_CACHE_ROOT = Path(dirpath)
            DOWNLOADS_CACHE_ROOT.mkdir(exist_ok=True, parents=True)
        elif dirkey == 'outputs-root':
            OUTPUTS_ROOT = Path(dirpath)
            OUTPUTS_ROOT.mkdir(exist_ok=True, parents=True)
    if DOWNLOADS_CACHE_ROOT is None:
        raise RuntimeError(f'Directory DOWNLOADS_CACHE_ROOT is not configured. Please configure it by adding the line downloads-cache-root=/path/to/directory/ to {LOCAL_CONFIG_ROOT}.')
    if OUTPUTS_ROOT is None:
        raise RuntimeError(f'Directory OUTPUTS_ROOT is not configured. Please configure it by adding the line outputs-root=/path/to/directory/ to {LOCAL_CONFIG_ROOT}.')

__all__ = [
    'ASCADV1_FIXED_ROOT', 'ASCADV1_VARIABLE_ROOT', 'ASCADV2_ROOT', 'CHES_CTF_2018_ROOT', 'DPAV4d2_ROOT',
    'DOWNLOADS_CACHE_ROOT', 'OUTPUTS_ROOT', 'append_directory_clargs', 'LOCAL_CONFIG_ROOT',
]