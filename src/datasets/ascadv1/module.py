from typing import *
import os
from copy import copy
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import lightning as L

from common import *
from .dataset import ASCADv1
from utils.calculate_dataset_stats import calculate_dataset_stats
from utils.download_unzip import download as _download, unzip, verify_sha256

FIXED_DOWNLOAD_URL = r'https://www.data.gouv.fr/s/resources/ascad/20180530-163000/ASCAD_data.zip'
FIXED_DOWNLOAD_SHA256 = r'a6884faf97133f9397aeb1af247dc71ab7616f3c181190f127ea4c474a0ad72c'
VARIABLE_DOWNLOAD_URL = r'https://static.data.gouv.fr/resources/ascad-atmega-8515-variable-key/20190903-083349/ascad-variable.h5'
VARIABLE_DOWNLOAD_SHA256 = r'd834da6ca5a288c4ba5add8e336845270a055d6eaf854dcf2f325a2eb6d7de06'
def download(root: str):
    version = root.split(os.sep)[-1].split('-')[-1]
    if version == 'fixed':
        url = FIXED_DOWNLOAD_URL
        hash = FIXED_DOWNLOAD_SHA256
    elif version == 'variable':
        url = VARIABLE_DOWNLOAD_URL
        hash = VARIABLE_DOWNLOAD_SHA256
    else:
        assert False
    dest_filename = url.split('/')[-1]
    dest_path = os.path.join(root, dest_filename)
    if os.path.exists(dest_path) and not(verify_sha256(dest_path, hash)):
        print(f'Download already exists at `{dest_path}`, but its SHA256 hash is incorrect. Deleting and re-downloading.')
        os.remove(dest_path)
    if not os.path.exists(dest_path):
        _download(url, dest_path, verbose=True)
        force_reextract = True
    else:
        print(f'Download already exists at `{dest_path}`.')
        force_reextract = False
    assert verify_sha256(dest_path, hash), 'Finished download, but the SHA256 hash is incorrect!'
    if version == 'fixed':
        unzip(dest_filename, root, overwrite=force_reextract)

class DataModule(L.LightningDataModule):
    def __init__(self,
        root: str,
        train_batch_size: int = 256,
        eval_batch_size: int = 2048,
        data_mean: Optional[Union[float, Sequence[float]]] = None,
        data_var: Optional[Union[float, Sequence[float]]] = None,
        dataset_kwargs: dict = {},
        dataloader_kwargs: dict = {}
    ):
        self.root = root
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.data_mean = data_mean
        self.data_var = data_var
        self.dataset_kwargs = dataset_kwargs
        self.dataloader_kwargs = dataloader_kwargs
        super().__init__()
    
    def setup(self, stage: str):
        self.profiling_dataset = ASCADv1(root=self.root, train=True, **self.dataset_kwargs)
        self.attack_dataset = ASCADv1(self.root, train=False, **self.dataset_kwargs)
        if (self.data_mean is None) or (self.data_var is None):
            self.data_mean, self.data_var = calculate_dataset_stats(self.profiling_dataset)
        self.data_mean, self.data_var = map(
            lambda x: torch.tensor(x, dtype=torch.float32) if isinstance(x, np.ndarray) else x.to(torch.float32), (self.data_mean, self.data_var)
        )
        basic_transform_mods = [
            transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
            transforms.Lambda(lambda x: (x - self.data_mean) / self.data_var.sqrt())
        ]
        transform = eval_transform = transforms.Compose(basic_transform_mods)
        target_transform = transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.long))
        self.profiling_dataset.transform = self.attack_dataset.transform = transform
        self.profiling_dataset.target_transform = self.attack_dataset.target_transform = target_transform
        self.train_indices = np.random.choice(len(self.profiling_dataset), int(0.9*len(self.profiling_dataset)), replace=False)
        self.val_indices = np.array([x for x in np.arange(len(self.profiling_dataset)) if not(x in self.train_indices)])
        self.val_dataset = Subset(copy(self.profiling_dataset), self.val_indices)
        self.train_dataset = Subset(copy(self.profiling_dataset), self.train_indices)
        if not 'num_workers' in self.dataloader_kwargs.keys():
            self.dataloader_kwargs['num_workers'] = max(os.cpu_count()//10, 1)
    
    def train_dataloader(self, override_batch_size=None):
        return DataLoader(
            self.train_dataset, shuffle=True, batch_size=self.train_batch_size if override_batch_size is None else override_batch_size, **self.dataloader_kwargs
        )
    
    def val_dataloader(self, override_batch_size=None):
        return DataLoader(
            self.val_dataset, shuffle=False, batch_size=self.eval_batch_size if override_batch_size is None else override_batch_size, **self.dataloader_kwargs
        )
    
    def test_dataloader(self, override_batch_size=None):
        return DataLoader(
            self.attack_dataset, shuffle=False, batch_size=self.eval_batch_size if override_batch_size is None else override_batch_size, **self.dataloader_kwargs
        )