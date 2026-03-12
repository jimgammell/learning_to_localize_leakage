import random
from math import ceil
from typing import Iterator, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import IterableDataset

from .base_dataset import Base_TorchDataset


class ChunkedDataset(IterableDataset):
    """Wraps a TorchDataset to read traces in sequential bulk chunks.

    Reads contiguous chunks via a single np_getitem(slice) call, converts
    to tensors once, and yields pre-formed batches. Use with
    DataLoader(batch_size=None) to skip auto-collation — batches come
    out ready to use.

    Chunk order is shuffled each epoch. Within each chunk, samples are
    shuffled and then sliced into batches.
    """
    def __init__(self, dataset: Base_TorchDataset, chunk_size: int = 2048,
                 batch_size: int = 256, shuffle: bool = True):
        self.dataset = dataset
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._length = len(dataset)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]]:
        self.dataset.init_data()
        n = self._length
        chunk_starts = list(range(0, n, self.chunk_size))

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            chunk_starts = chunk_starts[worker_info.id::worker_info.num_workers]

        if self.shuffle:
            random.shuffle(chunk_starts)

        for start in chunk_starts:
            end = min(start + self.chunk_size, n)
            # Single copy: memmap -> torch tensor directly
            trace = torch.from_numpy(
                np.array(self.dataset.traces[start:end, 2023:-2023])
            ).unsqueeze(1)  # (chunk, 1, timesteps)
            # Metadata is tiny, no performance concern
            idx = self.dataset.trace_indices[start:end]
            intermediate_variables = self.dataset.compute_intermediate_variables(
                self.dataset.keys[idx, :],
                self.dataset.plaintexts[idx, :],
                self.dataset.ciphertexts[idx, :]
            )
            target = np.concatenate([
                intermediate_variables[tv][..., self.dataset.config.target_byte]
                for tv in self.dataset.config.target_variable
            ], axis=-1)
            target = torch.as_tensor(target, dtype=torch.long)
            intermediate_variables = {
                k: torch.as_tensor(v, dtype=torch.long)
                for k, v in intermediate_variables.items()
            }
            if self.dataset.transform is not None:
                trace = self.dataset.transform(trace)
            if self.dataset.target_transform is not None:
                target = self.dataset.target_transform(target)

            chunk_len = end - start
            order = torch.randperm(chunk_len) if self.shuffle else torch.arange(chunk_len)

            # Yield pre-formed batches
            for batch_start in range(0, chunk_len, self.batch_size):
                batch_idx = order[batch_start:batch_start + self.batch_size]
                yield (
                    trace[batch_idx],
                    target[batch_idx],
                    {k: v[batch_idx] for k, v in intermediate_variables.items()}
                )

    def __len__(self) -> int:
        return ceil(self._length / self.batch_size)