import os

from common import *
from . import synthetic_aes, simple_gaussian, dpav4, aes_hd, ascadv1, ed25519_wolfssl, one_truth_prevails, nucleo

_DATASET_MODULES = {
    'synthetic-aes': synthetic_aes.module,
    'simple-gaussian': simple_gaussian,
    'dpav4': dpav4.module,
    'aes-hd': aes_hd.module,
    'ascadv1-fixed': ascadv1.module,
    'ascadv1-variable': ascadv1.module,
    'otiait': ed25519_wolfssl.module,
    'otp': one_truth_prevails.module,
    'nucleo': nucleo.module
}
AVAILABLE_DATASETS = list(_DATASET_MODULES.keys())

def _check_name(name):
    if not name in AVAILABLE_DATASETS:
        raise NotImplementedError(f'Unrecognized dataset name: {name}.')

def download(name, **kwargs):
    _check_name(name)
    _DATASET_MODULES[name].download(root=os.path.join(RESOURCE_DIR, name), **kwargs)

def load(name, **kwargs):
    _check_name(name)
    dataset_module = _DATASET_MODULES[name].DataModule(**kwargs)
    return dataset_module