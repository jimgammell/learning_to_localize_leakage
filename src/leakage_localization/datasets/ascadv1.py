from typing import Literal, Union, List, Optional, Callable, Tuple, Sequence, Dict, get_args
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray
import torch

from leakage_localization.utils import aes, get_sha256_hash
from .common import PARTITION
from .base_dataset import Base_NumpyDataset, Base_TorchDataset
from .compute_trace_statistics import compute_trace_statistics
from .convert_hdf5_to_binary import convert_hdf5_to_binary

TARGET_BYTE = Literal[
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
]
TARGET_VARIABLE = Literal[
    'subbytes',
    'r_in',
    'r',
    'r_out',
    'p__xor__k__xor__r_in',
    'subbytes__xor__r',
    'subbytes__xor__r_out',
    'key',
    'plaintext'
]

def repr_target(variable: TARGET_VARIABLE, byte: Optional[TARGET_BYTE] = None) -> str:
    assert variable in get_args(TARGET_VARIABLE)
    if byte is not None:
        assert byte in get_args(TARGET_BYTE)
    sbox_repr = r'\operatorname{Sbox}'
    k_repr = f'k_{byte}' if byte is not None else 'k'
    w_repr = f'w_{byte}' if byte is not None else 'w'
    r_repr = f'r_{byte}' if byte is not None else 'r'
    r_in_repr = r'r_{\mathrm{in}}'
    r_out_repr = r'r_{\mathrm{out}}'
    if variable == 'subbytes':
        rv = f'${sbox_repr}({w_repr} \\oplus {k_repr})$'
    elif variable == 'r_in':
        rv = f'${r_in_repr}$'
    elif variable == 'r':
        rv = f'${r_repr}$'
    elif variable == 'r_out':
        rv = f'${r_out_repr}$'
    elif variable == 'p__xor__k__xor__r_in':
        rv = f'${w_repr} \\oplus {k_repr} \\oplus {r_in_repr}$'
    elif variable == 'subbytes__xor__r':
        rv = f'${sbox_repr}({w_repr} \\oplus {k_repr}) \\oplus {r_repr}$'
    elif variable == 'subbytes__xor__r_out':
        rv = f'${sbox_repr}({w_repr} \\oplus {k_repr}) \\oplus {r_out_repr}$'
    elif variable == 'key':
        rv = f'${k_repr}$'
    elif variable == 'plaintext':
        rv = f'${w_repr}$'
    else:
        assert False
    return rv

@dataclass
class ASCADv1_Config:
    root: Union[str, Path]
    partition: PARTITION
    target_byte: Union[TARGET_BYTE, Sequence[TARGET_BYTE]] = 2
    target_variable: Union[TARGET_VARIABLE, Sequence[TARGET_VARIABLE]] = 'subbytes'
    variable_key: bool = False
    cropped_traces: bool = True

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
        assert isinstance(self.variable_key, bool)
        assert isinstance(self.cropped_traces, bool)

# wrapper around raw data without any PyTorch functionality
class ASCADv1_NumpyDataset(Base_NumpyDataset):
    def __init__(
            self,
            *,
            root: Union[str, Path],
            partition: PARTITION,
            target_byte: Union[TARGET_BYTE, List[TARGET_BYTE]] = 2,
            target_variable: Union[TARGET_VARIABLE, List[TARGET_VARIABLE]] = 'subbytes',
            variable_key: bool = False,
            cropped_traces: bool = False,
            binary_trace_file: bool = True,
    ):
        self.config = ASCADv1_Config(
            root=root,
            partition=partition,
            target_byte=target_byte,
            target_variable=target_variable,
            variable_key=variable_key,
            cropped_traces=cropped_traces
        )
        self.binary_trace_file = binary_trace_file

        if self.config.variable_key:
            if self.config.partition == 'profile':
                if self.binary_trace_file:
                    self.trace_indices = np.arange(200000)
                else:
                    self.trace_indices = np.concatenate([np.arange(0, 300000, 3), np.arange(1, 300000, 3)])
                    self.trace_indices.sort()
            elif self.config.partition == 'attack':
                if self.binary_trace_file:
                    self.trace_indices = np.arange(100000)
                else:
                    self.trace_indices = np.arange(2, 300000, 3)
            else:
                assert False
            if self.config.cropped_traces:
                self.timestep_count = 1400
                self.data_path = self.config.root / 'ascad-variable.h5'
                self.checksum = 'd834da6ca5a288c4ba5add8e336845270a055d6eaf854dcf2f325a2eb6d7de06'
            else:
                self.timestep_count = 250000
                self.data_path = self.config.root / 'atmega8515-raw-traces.h5'
                self.checksum = '6f13d7c380c937892c09b439910c4313d551adf011d2f4d76ad8b9b3de27b852'
        else:
            if self.config.partition == 'profile':
                self.trace_indices = np.arange(50000)
            elif self.config.partition == 'attack':
                if self.binary_trace_file:
                    self.trace_indices = np.arange(10000)
                else:
                    self.trace_indices = np.arange(50000, 60000)
            else:
                assert False
            if self.config.cropped_traces:
                self.timestep_count = 700
                self.data_path = self.config.root / 'ASCAD_data' / 'ASCAD_databases' / 'ASCAD.h5'
                self.checksum = 'f56625977fb6db8075ab620b1f3ef49a2a349ae75511097505855376e9684f91'
            else:
                self.timestep_count = 100000
                self.data_path = self.config.root / 'ASCAD_data' / 'ASCAD_databases' / 'ATMega8515_raw_traces.h5'
                self.checksum = '51e722f6c63a590ce2c4633c9a9534e8e1b22a9cde8e4532e32c11ac089d4625'
        if not self.data_path.exists():
            raise RuntimeError(f'Failed to find data file at {self.data_path}. Please follow instructions in README.md to download it.')
        checksum_passed_file = self.config.root / (self.data_path.name.split('.')[0] + '.checksum-passed')
        if not(checksum_passed_file.exists()):
            if get_sha256_hash(self.data_path) != self.checksum:
                raise RuntimeError(f'Checksum mismatch for file at {self.data_path}. File might be corrupt or incorrect.')
            with open(checksum_passed_file, 'w') as _:
                pass
        self.trace_count = len(self.trace_indices)

        self.traces = None
        self.keys = None
        self.plaintexts = None
        self.masks = None
        if self.binary_trace_file:
            self.binary_trace_filename = self.config.root / (self.data_path.name.split('.')[0] + f'.{self.config.partition}.dat')
            self.generate_trace_binary_file(use_progress_bar=True)
    
    def generate_trace_binary_file(self, use_progress_bar: bool = False):
        if not self.binary_trace_filename.exists():
            with h5py.File(self.data_path, 'r') as database:
                convert_hdf5_to_binary(database['traces'], self.binary_trace_filename, indices=self.trace_indices, chunk_size=4096, use_progress_bar=use_progress_bar)
    
    def init_data(self):
        if self.traces is None:
            assert self.binary_trace_filename.exists()
            self.traces = np.memmap(self.binary_trace_filename, dtype=np.int8, mode='r', shape=(self.trace_count, self.timestep_count), order='C')
        if self.keys is None or self.plaintexts is None or self.masks is None:
            assert self.data_path.exists()
            with h5py.File(self.data_path, 'r') as database:
                self.keys = np.array(database['metadata']['key'], dtype=np.uint8)
                self.plaintexts = np.array(database['metadata']['plaintext'], dtype=np.uint8)
                self.masks = np.array(database['metadata']['masks'], dtype=np.uint8)

    def get_trace_statistics(self, use_progress_bar: bool = False) -> Dict[str, NDArray[np.floating]]:
        cache_path = self.config.root / (self.data_path.name.split('.')[0] + '.stats-cache.npz')
        if not cache_path.exists():
            trace_mean, trace_var, trace_min, trace_max = compute_trace_statistics(self, chunk_size=4096, use_progress_bar=use_progress_bar)
            np.savez(cache_path, mean=trace_mean, var=trace_var, min=trace_min, max=trace_max)
        cache = np.load(cache_path, allow_pickle=True)
        return cache
    
    def compute_intermediate_variables(
            self,
            key: NDArray[np.uint8],
            plaintext: NDArray[np.uint8],
            masks: NDArray[np.uint8]
    ) -> Dict[str, NDArray[np.uint8]]:
        r = masks[..., self.config.target_byte]
        r_in = masks[..., -2, np.newaxis]
        r_out = masks[..., -1, np.newaxis]
        subbytes = aes.SBOX[key ^ plaintext]
        p__xor__k__xor__r_in = plaintext ^ key ^ r_in
        subbytes__xor__r = aes.SBOX[key ^ plaintext] ^ r
        subbytes__xor__r_out = aes.SBOX[key ^ plaintext] ^ r_out
        intermediate_variables = {
            'key': key,
            'plaintext': plaintext,
            'r': r,
            'r_in': r_in,
            'r_out': r_out,
            'subbytes': subbytes,
            'p__xor__k__xor__r_in': p__xor__k__xor__r_in,
            'subbytes__xor__r': subbytes__xor__r,
            'subbytes__xor__r_out': subbytes__xor__r_out
        }
        return intermediate_variables
    
    def __getitem__(self, _idx: Union[int, slice, NDArray[np.integer], Sequence[int]]) -> Tuple[NDArray[np.floating], NDArray[np.integer], Dict[str, NDArray[np.integer]]]:
        if isinstance(_idx, slice):
            _idx = np.arange(*_idx.indices(len(self.trace_indices)))
        elif isinstance(_idx, (int, np.integer)):
            _idx = np.array([_idx])
        elif not isinstance(_idx, np.ndarray):
            _idx = np.array(_idx)
        if not ((0 <= _idx).all() and (_idx < len(self.trace_indices)).all()):
            raise IndexError
        self.init_data()
        idx = self.trace_indices[_idx]
        trace = self.traces[idx, :]
        key = self.keys[idx, :]
        plaintext = self.plaintexts[idx, :]
        masks = self.masks[idx, :]
        intermediate_variables = self.compute_intermediate_variables(key, plaintext, masks)
        target = np.concatenate([
            intermediate_variables[target_variable][..., 0, np.newaxis] if target_variable in ['r_in', 'r_out']
            else intermediate_variables[target_variable][..., self.config.target_byte]
            for target_variable in self.config.target_variable
        ], axis=-2)
        return trace, target, intermediate_variables
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(' + ', '.join((
            f'partition={self.config.partition}',
            f'target_byte={self.config.target_byte}',
            f'target_variable={self.config.target_variable}',
            f'variable_key={self.config.variable_key}',
            f'cropped_traces={self.config.cropped_traces}'
        )) + ')'

class ASCADv1_TorchDataset(Base_TorchDataset, ASCADv1_NumpyDataset):
    def __init__(
            self,
            transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
            target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
            **dataset_config
    ):
        Base_TorchDataset.__init__(self, transform=transform, target_transform=target_transform)
        ASCADv1_NumpyDataset.__init__(self, **dataset_config)
    
    def __getitem__(self, _idx: Union[int, slice, NDArray[np.integer], Sequence[int]]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        trace, target, intermediate_variables = ASCADv1_NumpyDataset.__getitem__(self, _idx)
        trace = torch.from_numpy(trace).to(torch.float32)
        target = torch.from_numpy(target).to(torch.long)
        intermediate_variables = {k: torch.from_numpy(v).to(torch.long) for k, v in intermediate_variables.items()}
        if self.transform is not None:
            trace = self.transform(trace)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return trace, target, intermediate_variables