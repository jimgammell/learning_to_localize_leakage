# Adapted from https://github.com/ANSSI-FR/ASCAD/blob/master/ASCAD_generate.py

from typing import Union, Literal, Sequence, List, Optional, Dict, Tuple, Any, Callable, get_args
from pathlib import Path
from dataclasses import dataclass

import h5py
from tqdm import tqdm
import numpy as np
from numpy.typing import NDArray
import torch

from .common import PARTITION
from leakage_localization.utils import aes
from .base_dataset import Base_NumpyDataset, Base_TorchDataset
from .compute_trace_statistics import compute_trace_statistics

TARGET_BYTE = Literal[
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
]
TARGET_VARIABLE = Literal[
    'key',
    'plaintext',
    'alpha',
    'beta',
    'subbytes',
    'masked_subbytes'
]
G = np.array([0x0C, 0x05, 0x06, 0x0b, 0x09, 0x00, 0x0a, 0x0d, 0x03, 0x0e, 0x0f, 0x08, 0x04, 0x07, 0x01, 0x02], dtype=np.uint8)

def repr_target(variable: TARGET_VARIABLE, byte: Optional[TARGET_BYTE] = None) -> str:
    assert variable in get_args(TARGET_VARIABLE)
    if byte is not None:
        assert byte in get_args(TARGET_BYTE)
    sbox_repr = r'\operatorname{Sbox}'
    k_repr = f'k_{byte}' if byte is not None else 'k'
    w_repr = f'w_{byte}' if byte is not None else 'w'
    if variable == 'subbytes':
        rv = f'{sbox_repr}({w_repr} \\oplus {k_repr})'
    elif variable == 'masked_subbytes':
        rv = f'\\alpha \\times {sbox_repr}({w_repr} \\oplus {k_repr}) \\oplus \\beta'
    elif variable == 'beta':
        rv = r'\beta'
    elif variable == 'alpha':
        rv = r'\alpha'
    elif variable == 'key':
        rv = k_repr
    elif variable == 'plaintext':
        rv = w_repr
    else:
        assert False
    rv = f'${rv}$'
    return rv

def apply_perm_indices(i: NDArray[np.integer], m0: NDArray[np.integer], m1: NDArray[np.integer], m2: NDArray[np.integer], m3: NDArray[np.integer]) -> NDArray[np.integer]:
    x0 = m0 & 0x0F
    x1 = m1 & 0x0F
    x2 = m2 & 0x0F
    x3 = m3 & 0x0F
    perm_i = G[G[G[G[(15 - i)^x0]^x1]^x2]^x3]
    return perm_i

def extract_raw_dataset(datafiles: List[Path], trace_dest: Path, metadata_dest: Path, use_progress_bar: bool = False, exclude_indices: Optional[Sequence[int]] = None, chunk_size: int = 4096):
    if exclude_indices is None:
        exclude_indices = set()
    else:
        exclude_indices = set(exclude_indices)
    row_count = 0
    for datafile in datafiles:
        with h5py.File(datafile, 'r') as f:
            row_count += len(f['traces'])
    row_count -= len(exclude_indices)
    col_count = 1_000_000

    dest_memmap = np.memmap(trace_dest, mode='w+', dtype=np.int8, shape=(row_count, col_count), order='C')
    keys, plaintexts, masks = [], [], []
    if use_progress_bar:
        progress_bar = tqdm(total=row_count)
    memmap_idx = 0
    trace_idx = 0
    for datafile in datafiles:
        with h5py.File(datafile, mode='r') as datafile:
            n_traces = len(datafile['traces'])
            for chunk_start in range(0, n_traces, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_traces)
                chunk_global_indices = np.arange(trace_idx, trace_idx + (chunk_end - chunk_start))
                keep_mask = np.array([idx not in exclude_indices for idx in chunk_global_indices])
                n_keep = keep_mask.sum()
                if n_keep > 0:
                    trace_chunk = datafile['traces'][chunk_start:chunk_end][keep_mask]
                    dest_memmap[memmap_idx:memmap_idx + n_keep, :] = trace_chunk
                    keys.append(datafile['metadata']['key'][chunk_start:chunk_end][keep_mask])
                    plaintexts.append(datafile['metadata']['plaintext'][chunk_start:chunk_end][keep_mask])
                    masks.append(datafile['metadata']['masks'][chunk_start:chunk_end][keep_mask])
                    memmap_idx += n_keep
                trace_idx += chunk_end - chunk_start
                if use_progress_bar:
                    progress_bar.update(n_keep)
                dest_memmap.flush()
    assert memmap_idx == row_count
    np.savez(metadata_dest, keys=np.concatenate(keys), plaintexts=np.concatenate(plaintexts), masks=np.concatenate(masks))

@dataclass
class ASCADv2_Config:
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

class ASCADv2_NumpyDataset(Base_NumpyDataset):
    int_var_keys = get_args(TARGET_VARIABLE)

    def __init__(
            self,
            *,
            root: Union[str, Path],
            partition: PARTITION,
            target_byte: Union[TARGET_BYTE, List[TARGET_BYTE]] = 2,
            target_variable: Union[TARGET_VARIABLE, List[TARGET_VARIABLE]] = 'subbytes',
    ):
        self.config = ASCADv2_Config(
            root=root,
            partition=partition,
            target_byte=target_byte,
            target_variable=target_variable,
        )
        if self.config.partition == 'profile':
            self.trace_count = 399_997
        elif self.config.partition == 'attack':
            self.trace_count = 399_998
        else:
            assert False
        self.timestep_count = 1_000_000
        self.trace_indices = np.arange(self.trace_count)

        self.traces = None
        self.keys = None
        self.plaintexts = None
        self.masks = None
        self.binary_trace_path = self.config.root / f'ascadv2.{self.config.partition}.dat'
        self.metadata_path = self.config.root / f'ascadv2.{self.config.partition}.metadata.npz'
        if not self.binary_trace_path.exists() or not self.metadata_path.exists():
            datafiles = [
                self.config.root / f'ascadv2-stm32-conso-raw-traces{idx}.h5' for idx in range(1, 9)
            ]
            if self.config.partition == 'profile':
                datafiles = datafiles[:4]
                exclude_indices = {99_999, 199_999, 299_999}
            elif self.config.partition == 'attack':
                datafiles = datafiles[4:]
                exclude_indices = {99_999, 199_999}
            else:
                assert False
            extract_raw_dataset(datafiles, self.binary_trace_path, self.metadata_path, use_progress_bar=True, exclude_indices=exclude_indices)
    
    def init_data(self):
        if self.traces is None:
            assert self.binary_trace_path.exists()
            self.traces = np.memmap(self.binary_trace_path, dtype=np.int8, mode='r', shape=(self.trace_count, self.timestep_count), order='C')
        if self.keys is None or self.plaintexts is None or self.masks is None:
            assert self.metadata_path.exists()
            metadata = np.load(self.metadata_path, allow_pickle=True)
            self.keys = metadata['keys']
            self.plaintexts = metadata['plaintexts']
            self.masks = metadata['masks']
    
    # TODO: make sure this works when we simulate no permutation
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

    def get_trace_statistics(self, use_progress_bar: bool = False) -> Dict[str, NDArray[np.floating]]:
        assert self.config.partition == 'profile'
        cache_path = self.config.root / 'ascadv2.stats-cache.npz'
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
        perm_indices = apply_perm_indices(
            np.arange(16, dtype=np.uint8),
            masks[..., 0, np.newaxis],
            masks[..., 1, np.newaxis],
            masks[..., 2, np.newaxis],
            masks[..., 3, np.newaxis]
        )
        alpha = masks[..., 18, np.newaxis]
        beta = masks[..., 17, np.newaxis]
        subbytes_perm = aes.SBOX[key ^ plaintext]
        subbytes = np.take_along_axis(subbytes_perm, perm_indices, axis=-1)
        masked_subbytes = aes.mult_gf256(alpha, subbytes) ^ beta
        intermediate_variables = {
            'key': key,
            'plaintext': plaintext,
            'alpha': alpha,
            'beta': beta,
            'perm_indices': perm_indices,
            'subbytes_perm': subbytes_perm,
            'subbytes': subbytes,
            'masked_subbytes': masked_subbytes
        }
        return intermediate_variables

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
        masks = self.masks[idx, :]
        intermediate_variables = self.compute_intermediate_variables(key, plaintext, masks)
        target = np.concatenate([
            intermediate_variables[target_variable] if target_variable in {'alpha', 'beta', 'perm_indices'}
            else intermediate_variables[target_variable][..., self.config.target_byte]
            for target_variable in self.config.target_variable
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

class ASCADv2_TorchDataset(Base_TorchDataset, ASCADv2_NumpyDataset):
    def __init__(
            self,
            transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
            target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
            **dataset_config
    ):
        Base_TorchDataset.__init__(self, transform=transform, target_transform=target_transform)
        ASCADv2_NumpyDataset.__init__(self, **dataset_config)
    
    def __getitem__(self, _idx: Union[int, slice, NDArray[np.integer], Sequence[int]]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, NDArray[np.integer]]]:
        trace, target, intermediate_variables = ASCADv2_NumpyDataset.__getitem__(self, _idx)
        trace = torch.tensor(trace).unsqueeze(0)
        target = torch.tensor(target, dtype=torch.long)
        intermediate_variables = {k: torch.tensor(v, dtype=torch.long) for k, v in intermediate_variables.items()}
        if self.transform is not None:
            trace = self.transform(trace)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return trace, target, intermediate_variables
    
    def __len__(self) -> int:
        return ASCADv2_NumpyDataset.__len__(self)