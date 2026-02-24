# Following https://github.com/AISyLab/feature_selection_dlsca/blob/master/experiments/CHESCTF/generate_dataset.py
#  We use the files PinataAcqTask2.{1, 2, 3}_10k_upload.trs for profiling and PinataAcqTask2.4_10k_upload.trs for attacking.
#  There are 2 more files which only contain 1k traces each -- IDK what they are, but similarly to above implementation, I'm not using them.

from typing import Literal, Union, List, Sequence, Dict, Tuple, Any, Optional, Callable, get_args
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
import trsfile
from tqdm import tqdm
import torch

from .common import PARTITION
from .base_dataset import Base_NumpyDataset, Base_TorchDataset
from .compute_trace_statistics import compute_trace_statistics
from .convert_to_binary import convert_trs_to_binary
from leakage_localization.utils import aes

TARGET_BYTE = Literal[
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
]
TARGET_VARIABLE = Literal[
    'subbytes',
    'key',
    'plaintext',
    'ciphertext'
]

@dataclass
class CHESCTF2018_Config:
    root: Union[str, Path]
    partition: PARTITION
    target_byte: Union[TARGET_BYTE, Sequence[TARGET_BYTE]] = 2
    target_variable: Union[TARGET_VARIABLE, Sequence[TARGET_VARIABLE]] = 'subbytes'
    
    def __post_init__(self):
        if isinstance(self.root, str):
            self.root = Path(self.root)
        assert self.root.exists()
        assert self.partition in get_args(PARTITION)
        if isinstance(self.target_byte, int):
            self.target_byte = [self.target_byte]
        self.target_byte = np.array(self.target_byte)
        assert all(x in get_args(TARGET_BYTE) for x in self.target_byte)
        if isinstance(self.target_variable, str):
            self.target_variable = [self.target_variable]
        self.target_variable = np.array(self.target_variable)
        assert all(x in get_args(TARGET_VARIABLE) for x in self.target_variable)
        self.num_labels = len(self.target_byte)*len(self.target_variable)
        self.num_classes = 256

class CHESCTF2018_NumpyDataset(Base_NumpyDataset):
    int_var_keys = get_args(TARGET_VARIABLE)

    def __init__(
            self,
            *,
            root: Union[str, Path],
            partition: PARTITION,
            target_byte: Union[TARGET_BYTE, List[TARGET_BYTE]] = 0,
            target_variable: Union[TARGET_VARIABLE, List[TARGET_VARIABLE]] = 'subbytes',
    ):
        self.config = CHESCTF2018_Config(
            root=root,
            partition=partition,
            target_byte=target_byte,
            target_variable=target_variable,
        )

        if self.config.partition == 'profile':
            self.datanames = [
                r'PinataAcqTask2.1_10k_upload.trs',
                r'PinataAcqTask2.2_10k_upload.trs',
                r'PinataAcqTask2.3_10k_upload.trs',
            ]
            self.sample_offsets = [0, 0, 0] # the repo I'm referencing shifts the traces in the second file by 800 samples. However, an issue opened on their repo suggests not to do this, so I'm trying without.
            self.trace_count = 30_000
        elif self.config.partition == 'attack':
            self.datanames = [
                r'PinataAcqTask2.4_10k_upload.trs',
            ]
            self.trace_count = 10_000
            self.sample_offsets = [0]
        else:
            assert False
        self.timestep_count = 650_000

        self.binary_trace_path = self.config.root / f'ches_ctf_2018.{self.config.partition}.dat'
        if not self.binary_trace_path.exists():
            convert_trs_to_binary(
                [self.config.root / dataname for dataname in self.datanames],
                sample_offsets=self.sample_offsets,
                dest=self.binary_trace_path,
                use_progress_bar=True
            )
        self.metadata_path = self.config.root / f'ches_ctf_2018.{self.config.partition}.metadata.npz'
        if not self.metadata_path.exists():
            keys, plaintexts, ciphertexts = [], [], []
            for dataname in self.datanames:
                datapath = self.config.root / dataname
                with trsfile.open(datapath, 'r') as data_file:
                    for x in data_file:
                        plaintext, ciphertext, key = np.split(np.frombuffer(x.data, dtype=np.uint8, count=48), 3)
                        plaintexts.append(plaintext)
                        ciphertexts.append(ciphertext)
                        keys.append(key)
            keys = np.stack(keys).astype(np.uint8)
            plaintexts = np.stack(plaintexts).astype(np.uint8)
            ciphertexts = np.stack(ciphertexts).astype(np.uint8)
            np.savez(self.metadata_path, keys=keys, plaintexts=plaintexts, ciphertexts=ciphertexts)
        self.traces = None # will load lazily to avoid multiprocessing-related issues
        self.keys = None
        self.plaintexts = None
        self.ciphertexts = None
        self.trace_indices = np.arange(self.trace_count)

    def init_data(self):
        if self.traces is None:
            self.traces = np.memmap(self.binary_trace_path, dtype=np.int8, mode='r', shape=(self.trace_count, self.timestep_count), order='C')
        if self.keys is None or self.plaintexts is None or self.ciphertexts is None:
            metadata = np.load(self.metadata_path, allow_pickle=True)
            self.keys = metadata['keys']
            self.plaintexts = metadata['plaintexts']
            self.ciphertexts = metadata['ciphertexts']
    
    def get_trace_statistics(self, use_progress_bar: bool = False) -> Dict[str, NDArray[np.floating]]:
        assert self.config.partition == 'profile'
        cache_path = self.config.root / 'ches_ctf_2018.stats-cache.npz'
        if not cache_path.exists():
            trace_mean, trace_var, trace_min, trace_max = compute_trace_statistics(self, chunk_size=4096, use_progress_bar=use_progress_bar)
            np.savez(cache_path, mean=trace_mean, var=trace_var, min=trace_min, max=trace_max)
        cache = np.load(cache_path, allow_pickle=True)
        return cache
    
    def compute_intermediate_variables(
            self,
            key: NDArray[np.uint8],
            plaintext: NDArray[np.uint8],
            ciphertext: NDArray[np.uint8]
    ) -> Dict[str, NDArray[np.uint8]]:
        subbytes = aes.SBOX[key ^ plaintext]
        intermediate_variables = {
            'key': key,
            'plaintext': plaintext,
            'ciphertext': ciphertext,
            'subbytes': subbytes
        }
        return intermediate_variables
    
    @staticmethod
    def target_preds_to_key_preds(
        target_preds: NDArray[np.floating],
        intermediate_variables: Dict[str, NDArray[np.uint8]]
    ) -> NDArray[np.floating]:
        plaintext = intermediate_variables['plaintext']
        key_candidates = np.arange(256, dtype=np.uint8)
        target_indices = aes.SBOX[key_candidates ^ plaintext[..., np.newaxis]]
        key_preds = np.take_along_axis(target_preds, target_indices.astype(np.intp), axis=-1)
        return key_preds

    def np_getitem(self, _idx: Union[int, slice, NDArray[np.integer], Sequence[int]]) -> Tuple[NDArray[np.floating], NDArray[np.integer], Dict[str, NDArray[np.integer]]]:
        if isinstance(_idx, slice):
            _idx = np.arange(*_idx.indices(len(self.trace_indices)))
        elif isinstance(_idx, (int, np.integer)):
            _idx = np.array([_idx])
        elif not isinstance(_idx, np.ndarray):
            _idx = np.array(_idx)
        if not ((0 <= _idx).all() and (_idx < len(self.trace_indices)).all()):
            raise IndexError
        if len(_idx) == 1:
            _idx = _idx[0]
        self.init_data()
        idx = self.trace_indices[_idx]
        trace = self.traces[idx, :]
        key = self.keys[idx, :]
        plaintext = self.plaintexts[idx, :]
        ciphertext = self.ciphertexts[idx, :]
        intermediate_variables = self.compute_intermediate_variables(key, plaintext, ciphertext)
        target = np.concatenate([
            intermediate_variables[target_variable][..., self.config.target_byte] for target_variable in self.config.target_variable
        ], axis=-1)
        return trace, target, intermediate_variables

    def __getitem__(self, *args, **kwargs) -> Any:
        return self.np_getitem(*args, **kwargs)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(' + ', '.join((
            f'partition={self.config.partition}',
            f'target_byte={self.config.target_byte}',
            f'target_variable={self.config.target_variable}'
        )) + ')'

class CHESCTF2018_TorchDataset(Base_TorchDataset, CHESCTF2018_NumpyDataset):
    def __init__(
            self,
            transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
            target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
            **dataset_config
    ):
        Base_TorchDataset.__init__(self, transform=transform, target_transform=target_transform)
        CHESCTF2018_NumpyDataset.__init__(self, **dataset_config)
    
    def __getitem__(self, _idx: Union[int, slice, NDArray[np.integer], Sequence[int]]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, NDArray[np.integer]]]:
        trace, target, intermediate_variables = CHESCTF2018_NumpyDataset.__getitem__(self, _idx)
        trace = torch.tensor(trace).unsqueeze(0)
        target = torch.tensor(target, dtype=torch.long)
        intermediate_variables = {k: torch.tensor(v, dtype=torch.long) for k, v in intermediate_variables.items()}
        if self.transform is not None:
            trace = self.transform(trace)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return trace, target, intermediate_variables
    
    def __len__(self) -> int:
        return CHESCTF2018_NumpyDataset.__len__(self)