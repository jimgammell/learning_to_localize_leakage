# Adapted from https://github.com/AISyLab/feature_selection_dlsca/blob/master/experiments/DPAV42

from typing import Literal, Union, Sequence, List, Dict, Tuple, Any, Optional, Callable, get_args
from pathlib import Path
from dataclasses import dataclass

from tqdm import tqdm
import numpy as np
from numpy.typing import NDArray
import bz2
import torch

from .common import PARTITION
from .base_dataset import Base_NumpyDataset, Base_TorchDataset
from .compute_trace_statistics import compute_trace_statistics
from leakage_localization.utils import aes

TARGET_BYTE = Literal[
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
]
TARGET_VARIABLE = Literal[
    'subbytes',
    'plaintext',
    'ciphertext',
    'mask',
    'key',
]
MASK = np.array([
    3, 12, 53, 58, 80, 95, 102, 105, 150, 153, 160, 175, 197, 202, 243, 252
], dtype=np.uint8)

@dataclass
class DPAv4d2_Config:
    root: Union[str, Path]
    partition: PARTITION
    target_byte: Union[TARGET_BYTE, Sequence[TARGET_BYTE]]
    target_variable: Union[TARGET_VARIABLE, Sequence[TARGET_VARIABLE]]

    def __post_init__(self):
        if isinstance(self.root, str):
            self.root = Path(self.root)
        assert self.root.exists()
        assert self.partition in get_args(PARTITION)
        if isinstance(self.target_byte, int):
            self.target_byte = [self.target_byte]
        self.target_byte = np.array(self.target_byte)
        if isinstance(self.target_variable, str):
            self.target_variable = [self.target_variable]
        self.target_variable = np.array(self.target_variable)
        assert all(x in get_args(TARGET_VARIABLE) for x in self.target_variable)
        self.num_labels = len(self.target_byte)*len(self.target_variable)
        self.num_classes = 256

# not making this generic since it's pretty specific to this particular dataset
def prepare_dataset(root: Path, partition: PARTITION):
    if partition == 'profile':
        indices = np.arange(0, 75_000)
    elif partition == 'attack':
        indices = np.arange(75_000, 80_000)
    else:
        assert False
    row_count = len(indices)
    col_count = 1_704_046
    plaintexts = np.empty((row_count, 16), dtype=np.uint8)
    ciphertexts = np.empty((row_count, 16), dtype=np.uint8)
    masks = np.empty((row_count, 16), dtype=np.uint8)
    keys = np.empty((row_count, 16), dtype=np.uint8)
    with open(root / 'v4_2' / 'dpav4_2_index', 'r') as index_file:
        progress_bar = tqdm(total=row_count, desc='Metadata extraction')
        idx = 0
        for line_idx, line in enumerate(index_file.readlines()):
            if line_idx < indices[0] or line_idx > indices[-1]:
                continue
            key = np.frombuffer(bytearray.fromhex(line[0:32]), dtype=np.uint8)
            plaintext = np.frombuffer(bytearray.fromhex(line[33:65]), dtype=np.uint8)
            ciphertext = np.frombuffer(bytearray.fromhex(line[66:98]), dtype=np.uint8)
            offset1 = [int(s, 16) for s in line[99:115]]
            offset2 = [int(s, 16) for s in line[116:132]]
            offset3 = [int(s, 16) for s in line[133:149]]
            keys[idx, :] = key
            plaintexts[idx, :] = plaintext
            ciphertexts[idx, :] = ciphertext
            for byte in range(16):
                masks[idx, byte] = int(MASK[int(offset3[byte] + 1) % 16])
            idx += 1
            progress_bar.update(1)
    np.savez(root / f'metadata.{partition}.npz', plaintexts=plaintexts, ciphertexts=ciphertexts, masks=masks, keys=keys)
    
    traces = np.memmap(root / f'traces.{partition}.dat', shape=(row_count, col_count), dtype=np.int8, mode='w+', order='C')
    progress_bar = tqdm(total=row_count, desc='Trace extraction')
    idx = 0
    for key in range(16):
        for file_idx in range(5000*key, 5000*(key+1)):
            if file_idx < indices[0] or file_idx > indices[-1]:
                continue
            trace_filename = f'DPACV42_{file_idx:06}.trc.bz2'
            trace_path = root / 'v4_2' / 'DPA_contestv4_2' / f'k{key:02}' / trace_filename
            trace = np.frombuffer(bz2.BZ2File(trace_path).read()[357:-357], dtype=np.int8)
            traces[idx, :] = trace
            idx += 1
            progress_bar.update(1)
        traces.flush()

class DPAv4d2_NumpyDataset(Base_NumpyDataset):
    int_var_keys = get_args(TARGET_VARIABLE)

    def __init__(
            self,
            *,
            root: Union[str, Path],
            partition: PARTITION,
            target_byte: Union[TARGET_BYTE, List[TARGET_BYTE]] = 0,
            target_variable: Union[TARGET_VARIABLE, List[TARGET_VARIABLE]] = 'subbytes'
    ):
        self.config = DPAv4d2_Config(
            root=root,
            partition=partition,
            target_byte=target_byte,
            target_variable=target_variable
        )
        if self.config.partition == 'profile':
            self.trace_count = 75_000
        elif self.config.partition == 'attack':
            self.trace_count = 5_000
        else:
            assert False
        self.timestep_count = 1_700_000 # actually 1_704_046, but I'm going to crop it down so it's divisible by more patch sizes
        
        self.binary_trace_path = self.config.root / f'traces.{self.config.partition}.dat'
        self.metadata_path = self.config.root / f'metadata.{self.config.partition}.npz'
        if not self.binary_trace_path.exists() and self.metadata_path.exists():
            prepare_dataset(self.config.root, self.config.partition)
        
        self.traces = None
        self.keys = None
        self.plaintexts = None
        self.ciphertexts = None
        self.trace_indices = np.arange(self.trace_count)
    
    def init_data(self):
        if self.traces is None:
            self.traces = np.memmap(self.binary_trace_path, dtype=np.int8, mode='r', shape=(self.trace_count, 1_704_046), order='C')
        if self.keys is None or self.plaintexts is None or self.ciphertexts is None:
            metadata = np.load(self.metadata_path, allow_pickle=True)
            self.keys = metadata['keys']
            self.plaintexts = metadata['plaintexts']
            self.ciphertexts = metadata['ciphertexts']
    
    def get_trace_statistics(self, use_progress_bar: bool = False) -> Dict[str, NDArray[np.floating]]:
        assert self.config.partition == 'profile'
        cache_path = self.config.root / 'stats-cache.npz'
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
        trace = self.traces[idx, 2023:-2023]
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

class DPAv4d2_TorchDataset(Base_TorchDataset, DPAv4d2_NumpyDataset):
    def __init__(
            self,
            transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
            target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
            **dataset_config
    ):
        Base_TorchDataset.__init__(self, transform=transform, target_transform=target_transform)
        DPAv4d2_NumpyDataset.__init__(self, **dataset_config)
    
    def __getitem__(self, _idx: Union[int, slice, NDArray[np.integer], Sequence[int]]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, NDArray[np.integer]]]:
        trace, target, intermediate_variables = DPAv4d2_NumpyDataset.__getitem__(self, _idx)
        trace = torch.tensor(trace).unsqueeze(0)
        target = torch.tensor(target, dtype=torch.long)
        intermediate_variables = {k: torch.tensor(v, dtype=torch.long) for k, v in intermediate_variables.items()}
        if self.transform is not None:
            trace = self.transform(trace)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return trace, target, intermediate_variables
    
    def __len__(self) -> int:
        return DPAv4d2_NumpyDataset.__len__(self)