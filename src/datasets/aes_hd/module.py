from typing import *
import os
import shutil
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import lightning as L

from common import *
from .dataset import AES_HD
from utils.calculate_dataset_stats import calculate_dataset_stats
from utils.download_unzip import download as _download, unzip, verify_sha256

# Downloading the Zaid version, since the official version appears to no longer be available.
DOWNLOAD_URL = r'https://github.com/gabzai/Methodology-for-efficient-CNN-architectures-in-SCA/raw/refs/heads/master/AES_HD/AES_HD_dataset.zip'
DOWNLOAD_SHA256 = r'00a3d02f01bae8c4fcefda33e3d1adb57bed0509ded3cdcf586e213b3d87e41b'
def download(root: str):
    dest_filename = DOWNLOAD_URL.split('/')[-1]
    dest_path = os.path.join(root, dest_filename)
    if os.path.exists(dest_path) and not(verify_sha256(dest_path, DOWNLOAD_SHA256)):
        print(f'Download already exists at `{dest_path}`, but its SHA256 hash is incorrect. Deleting and re-downloading.')
        os.remove(dest_path)
    if not os.path.exists(dest_path):
        _download(DOWNLOAD_URL, dest_path, verbose=True)
        force_reextract = True
    else:
        print(f'Download already exists at `{dest_path}`.')
        force_reextract = False
    assert verify_sha256(dest_path, DOWNLOAD_SHA256), 'Finished download, but the SHA256 hash is incorrect!'
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
        self.profiling_dataset = AES_HD(root=self.root, train=True)
        self.attack_dataset = AES_HD(self.root, train=False)
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
        self.val_dataset = Subset(self.profiling_dataset, self.val_indices)
        self.train_dataset = Subset(self.profiling_dataset, self.train_indices)
        if not 'num_workers' in self.dataloader_kwargs.keys():
            self.dataloader_kwargs['num_workers'] = os.cpu_count()//10
    
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