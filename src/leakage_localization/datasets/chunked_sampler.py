from typing import Optional, Iterator

import torch
from torch.utils.data import Dataset, Sampler

class ChunkedSampler(Sampler[int]):
    def __init__(
            self,
            dataset: Dataset,
            *,
            chunk_size: int,
            shuffle: bool,
            seed: Optional[int] = None
    ):
        super().__init__()
        self.dataset = dataset
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
    
    def set_epoch(self, epoch: int):
        self.epoch = epoch
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __iter__(self) -> Iterator[int]:
        chunk_start_indices = list(range(0, len(self.dataset), self.chunk_size))
        if self.shuffle:
            g = torch.Generator()
            if self.seed is not None:
                g.manual_seed(self.seed + self.epoch)
            perm = torch.randperm(len(chunk_start_indices), generator=g).tolist()
            chunk_start_indices = [chunk_start_indices[idx] for idx in perm]
        for chunk_start_idx in chunk_start_indices:
            chunk_end_idx = min(chunk_start_idx + self.chunk_size, len(self.dataset))
            chunk_indices = list(range(chunk_start_idx, chunk_end_idx))
            if self.shuffle:
                perm = torch.randperm(len(chunk_indices), generator=g).tolist()
                chunk_indices = [chunk_indices[idx] for idx in perm]
            yield from chunk_indices
        self.epoch += 1