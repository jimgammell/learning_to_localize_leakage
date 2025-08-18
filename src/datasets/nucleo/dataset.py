from typing import Callable, Optional, Any
import os
from random import randint

import numpy as np
from numba import jit
from torch.utils.data import Dataset
import estraces

from common import *
from utils.aes import AES_SBOX

INDICES = np.random.default_rng(seed=0x18273645).permutation(100000)
HW_OF_ID = np.array([bin(i).count('1') for i in range(256)], dtype=np.int64)
CLASS_SIZES = np.bincount(HW_OF_ID, minlength=9)
LOG_CLASS_SIZES = np.log(CLASS_SIZES.astype(np.float32))

@jit(nopython=True)
def to_key_preds(hw_logits, plaintext, constants):
    hw_logits = hw_logits.reshape(-1, 9)
    id_logits = hw_logits = LOG_CLASS_SIZES
    id_logits = id_logits[:, HW_OF_ID]
    id_logits_max = np.max(id_logits, axis=-1, keepdims=True)
    id_logits = id_logits - id_logits_max
    logsumexp = np.log(np.sum(np.exp(id_logits), axis=-1, keepdims=True))
    id_logp = id_logits - logsumexp
    key_preds = id_logp[AES_SBOX[np.arange(256, dtype=np.uint8) ^ plaintext]]
    return key_preds

class Nucleo(Dataset):
    def __init__(
            self, root: str, desync_level: int = 10, train: bool = True, target_byte: int = 0, hw_targets: bool = True,
            transform: Optional[Callable[[Any], torch.Tensor]] = None, target_transform: Optional[Callable[[Any], torch.Tensor]] = None
    ):
        super().__init__()
        if train:
            self.indices = INDICES[:-20000]
        else:
            self.indices = INDICES[-20000:]
        self.target_byte = target_byte
        self.hw_targets = hw_targets
        dataset = estraces.read_ths_from_ets_file(os.path.join(root, r'Nucleo_AES_masked_non_shuffled.ets'))
        self.traces = np.array(dataset.samples)[self.indices, :]
        self.keys = np.array(dataset.key)[self.indices, self.target_byte]
        self.plaintexts = np.array(dataset.plaintext)[self.indices, self.target_byte]
        self.masks = np.array(dataset.mask)[self.indices, :]
        self.targets = AES_SBOX[self.keys ^ self.plaintexts]
        if self.hw_targets:
            self.class_count = 9
        else:
            self.class_count = 256
        self.desync_level = desync_level
        self.transform = transform
        self.target_transform = target_transform
        self.timesteps_per_trace = self.traces.shape[-1]
        self.return_metadata = False
    
    def configure_target(self, target: Literal['id', 'hw']):
        if target == 'id':
            self.hw_targets = False
            self.class_count = 256
        elif target == 'hw':
            self.hw_targets = True
            self.class_count = 9
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        trace = self.traces[idx, np.newaxis, :]
        target = self.targets[idx]
        if self.hw_targets:
            target = np.unpackbits(target).sum()
        if self.desync_level > 0:
            shift = randint(0, self.desync_level)
            trace = np.roll(trace, shift, axis=-1)
        if self.transform is not None:
            trace = self.transform(trace)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.return_metadata:
            metadata = dict()
            metadata['label'] = target
            metadata['subbytes'] = target
            metadata['subbytes__r0'] = target ^ self.masks[idx, 0]
            metadata['subbytes__r1'] = target ^ self.masks[idx, 1]
            metadata['r0__k__pt'] = self.masks[idx, 0] ^ self.keys[idx] ^ self.plaintexts[idx]
            metadata['r0'] = self.masks[idx, 0]
            metadata['r1'] = self.masks[idx, 1]
            return trace, target, metadata
        else:
            return trace, target
    
    def __len__(self) -> int:
        return len(self.traces)