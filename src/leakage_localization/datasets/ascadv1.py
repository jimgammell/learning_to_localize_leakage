from typing import Literal, Union, List, Optional, Callable, get_args
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from leakage_localization.utils import aes, get_sha256_hash

PARTITIONS = Literal['profile', 'attack']
TARGET_VARIABLES = Literal[
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

@dataclass
class ASCADv1_Config:
    root: Union[str, Path]
    partition: PARTITIONS
    target_byte: Union[int, List[int]] = 2
    target_variable: Union[TARGET_VARIABLES, List[TARGET_VARIABLES]] = 'subbytes'
    variable_key: bool = False
    cropped_traces: bool = True

    def __post_init__(self):
        if isinstance(self.root, str):
            self.root = Path(self.root)
        assert self.root.exists()
        assert self.partition in get_args(PARTITIONS)
        if isinstance(self.target_byte, int):
            self.target_byte = [self.target_byte]
        else:
            assert isinstance(self.target_byte, list)
        assert all(isinstance(x, int) and 0 <= x < 16 for x in self.target_byte)
        if isinstance(self.target_variable, str):
            self.target_variable = [self.target_variable]
        else:
            assert isinstance(self.target_variable, list)
        assert all(x in get_args(TARGET_VARIABLES) for x in self.target_variable)
        assert isinstance(self.variable_key, bool)
        assert isinstance(self.cropped_traces, bool)

class ASCADv1(Dataset):
    def __init__(
            self,
            *,
            root: Path,
            partition: PARTITIONS,
            target_byte: Union[int, List[int]] = 2,
            target_variable: Union[TARGET_VARIABLES, List[TARGET_VARIABLES]] = 'subbytes',
            variable_key: bool = False,
            cropped_traces: bool = False,
            transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
            target_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None
    ):
        super().__init__()

        self.config = ASCADv1_Config(
            root=root,
            partition=partition,
            target_byte=target_byte,
            target_variable=target_variable,
            variable_key=variable_key,
            cropped_traces=cropped_traces
        )
        self.transform = transform
        self.target_transform = target_transform

        if self.config.variable_key:
            if self.config.partition == 'profile':
                self.trace_indices = np.concatenate([np.arange(0, 300000, 3), np.arange(1, 300000, 3)])
            elif self.config.partition == 'attack':
                self.trace_indices = np.arange(2, 300000, 3)
            else:
                assert False
            if self.config.cropped_traces:
                self.timestep_count = 1400
                self.data_path = self.config.root / 'ascad-variable.h5'
                self.checksum = 'd834da6ca5a288c4ba5add8e336845270a055d6eaf854dcf2f325a2eb6d7de06'
            else:
                self.timestep_indices = 250000
                self.data_path = self.config.root / 'atmega8515-raw-traces.h5'
                self.checksum = self.config.root / '6f13d7c380c937892c09b439910c4313d551adf011d2f4d76ad8b9b3de27b852'
        else:
            if self.config.partition == 'profile':
                self.trace_indices = np.arange(50000)
            elif self.config.partition == 'attack':
                self.trace_indices = np.arange(50000, 60000)
            else:
                assert False
            if self.config.cropped_traces:
                self.timestep_count = 700
                self.data_path = self.config.root / 'ASCAD_data' / 'ASCAD_databases' / 'ASCAD.h5'
                if not self.data_path.exists():
                    raise RuntimeError(f'Failed to find ASCADv1-fixed-cropped data file at {self.data_path}. Please follow instructions in README.md to download it.')
                self.checksum = 'f56625977fb6db8075ab620b1f3ef49a2a349ae75511097505855376e9684f91'
            else:
                self.timestep_count = 100000
                self.data_path = self.config.root / 'ASCAD_data' / 'ASCAD_databases' / 'ATMega8515_raw_traces.h5'
                self.checksum = '51e722f6c63a590ce2c4633c9a9534e8e1b22a9cde8e4532e32c11ac089d4625'
        self.trace_count = len(self.trace_indices)
    
    def __len__(self) -> int:
        return self.trace_count