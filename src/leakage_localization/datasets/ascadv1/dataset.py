import os
import h5py
import numpy as np
from numba import jit
import torch
from torch.utils.data import Dataset

from utils.aes import *

BYTE_ORDER = np.array([15, 12, 13, 1, 8, 10, 0, 3, 7, 6, 9, 5, 11, 2, 4, 14])
# The order in which the SBox operation is applied to the different key bytes.
#  Useful for computing S_{prev} and the security load, as suggested by Egger (2021).

@jit(nopython=True)
def to_key_preds(int_var_preds, plaintext, constants=None):
    return int_var_preds[AES_SBOX[np.arange(256, dtype=np.uint8) ^ plaintext]]

class ASCADv1(Dataset):
    def __init__(self,
        root=None,
        train=True,
        target_byte=2,
        target_values='subbytes',
        desync=0,
        variable_keys=False,
        raw_traces=False,
        transform=None,
        target_transform=None
    ):
        super().__init__()
        self.root = root
        self.train = train
        self.target_byte = np.arange(16) if target_byte == 'all' else np.array([target_byte]) if not hasattr(target_byte, '__len__') else target_byte
        assert self.target_byte == 2
        self.prev_byte = np.array([11]) # TODO: should make this more general in case people want to check other bytes
        self.target_values = [target_values] if isinstance(target_values, str) else target_values
        self.desync = desync
        self.variable_keys = variable_keys
        self.raw_traces = raw_traces
        self.transform = transform
        self.target_transform = target_transform
        self.return_metadata = False
        if raw_traces:
            raise NotImplementedError
        
        self.construct()
    
    def construct(self):
        if self.variable_keys:
            self.data_path = os.path.join(self.root, f'ascad-variable-desync{self.desync}.h5' if self.desync > 0 else 'ascad-variable.h5')
            if self.train:
                self.data_indices = np.arange(0, 200000)
            else:
                self.data_indices = np.arange(0, 100000)
        else:
            self.data_path = os.path.join(self.root, 'ASCAD_data', 'ASCAD_databases', 'ASCAD' + (f'desync{self.desync}' if self.desync != 0 else '') + '.h5')
            if self.train:
                self.data_indices = np.arange(0, 50000)
            else:
                self.data_indices = np.arange(0, 10000)
        self.dataset_length = len(self.data_indices)
        self.traces, self.metadata = self._load_datapoints_from_disk(self.data_indices)
        eg_trace, _ = self.load_datapoints(0)
        self.data_shape = eg_trace.shape
        self.timesteps_per_trace = self.data_shape[-1]
        self.class_count = 256
    
    def _load_datapoints_from_disk(self, indices):
        with h5py.File(self.data_path) as _database_file:
            if self.train:
                database_file = _database_file['Profiling_traces']
            else:
                database_file = _database_file['Attack_traces']
            traces = np.array(database_file['traces'][indices, :], dtype=np.int8)
            if traces.ndim == 1:
                traces = traces[np.newaxis, :].astype(np.float32)
            else:
                traces = traces[:, np.newaxis, :].astype(np.float32)
            metadata = {
                'plaintext': np.array(database_file['metadata']['plaintext'][indices, :], dtype=np.uint8),
                'key': np.array(database_file['metadata']['key'][indices, :], dtype=np.uint8),
                'masks': np.array(database_file['metadata']['masks'][indices], dtype=np.uint8)
            }
        return traces, metadata
    
    def _load_datapoints_from_ram(self, indices):
        traces = self.traces[indices, :, :]
        metadata = {key: val[indices] for key, val in self.metadata.items()}
        return traces, metadata
    
    def load_datapoints(self, indices):
        return self._load_datapoints_from_ram(indices)
    
    def compute_target(self, metadata):
        key = metadata['key'][..., self.target_byte]
        plaintext = metadata['plaintext'][..., self.target_byte]
        key_prev = metadata['key'][..., self.prev_byte]
        plaintext_prev = metadata['plaintext'][..., self.prev_byte]
        masks = metadata['masks']
        if key.ndim > 1:
            batch_size = key.shape[0]
        else:
            batch_size = 1
            key = np.array([key])
            plaintext = np.array([plaintext])
            key_prev = np.array([key_prev])
            plaintext_prev = np.array([plaintext_prev])
            masks = masks[np.newaxis, :]
        assert batch_size == key.shape[0] == plaintext.shape[0] == masks.shape[0] == key_prev.shape[0] == plaintext_prev.shape[0]
        r_in = masks[:, -2].squeeze()[..., np.newaxis]
        r_out = masks[:, -1].squeeze()[..., np.newaxis]
        if not self.variable_keys:
            rr = np.concatenate([np.zeros((batch_size, 2), dtype=np.uint8), masks[:, :-2]], axis=1)
        else:
            rr = masks[:, :-2]
        r = rr[..., self.target_byte].squeeze()[..., np.newaxis]
        r_prev = rr[..., self.prev_byte].squeeze()[..., np.newaxis]
        subbytes = AES_SBOX[key ^ plaintext]
        subbytes__r = subbytes ^ r
        subbytes__r_out = subbytes ^ r_out
        p__k__r_in = plaintext ^ key ^ r_in
        S_prev = AES_SBOX[key_prev ^ plaintext_prev] ^ r_prev
        S_prev__subbytes__r_out = S_prev ^ subbytes ^ r_out
        security_load = AES_SBOX[S_prev ^ r_in] ^ r_out
        aux_metadata = { # the latter were identified as the 'leaky' random shares by Egger (2021)
            'subbytes': subbytes,
            'r_in': r_in,
            'r': r,
            'r_out': r_out,
            'plaintext__key__r_in': p__k__r_in,
            'subbytes__r': subbytes__r,
            'subbytes__r_out': subbytes__r_out,
            's_prev__subbytes__r_out': S_prev__subbytes__r_out,
            'security_load': security_load,
            'key': key.squeeze(),
            'plaintext': plaintext.squeeze(),
            'key_prev': key_prev.squeeze(),
            'plaintext_prev': plaintext_prev.squeeze()
        }
        targets = []
        for target_val in self.target_values:
            targets.append(aux_metadata[target_val].squeeze())
        if batch_size > 1:
            targets = np.stack(targets, axis=1)
        else:
            targets = np.stack(targets).squeeze()
        return targets, aux_metadata
    
    def __getitem__(self, indices):
        indices = self.data_indices[indices]
        trace, metadata = self.load_datapoints(indices)
        target, metadata = self.compute_target(metadata)
        metadata.update({'label': target})
        if self.transform is not None:
            trace = self.transform(trace)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.return_metadata:
            return trace, target, metadata
        else:
            return trace, target
    
    def __len__(self):
        return len(self.data_indices)