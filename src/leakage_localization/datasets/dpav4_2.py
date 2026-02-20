# Adapted from https://github.com/AISyLab/feature_selection_dlsca/blob/master/experiments/DPAV42

from typing import Literal, Sequence
from pathlib import Path

from tqdm import tqdm
import numpy as np
import bz2

from .common import PARTITION

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
            trace_filename = f'DPACV42_{file_idx:06}.trc.bz2'
            trace_path = root / 'v4_2' / 'DPA_contestv4_2' / f'k{key:02}' / trace_filename
            trace = np.frombuffer(bz2.BZ2File(trace_path).read()[357:-357], dtype=np.int8)
            traces[idx, :] = trace
            idx += 1
            progress_bar.update(1)
        traces.flush()