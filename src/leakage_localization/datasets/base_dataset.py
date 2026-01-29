from typing import Callable, Optional, Dict, Union, Sequence, Tuple
from abc import ABC, abstractmethod

import h5py
import numpy as np
from numpy.typing import NDArray
import torch
from torch.utils.data import Dataset

class Base_NumpyDataset(ABC):
    database: Optional[h5py.File]
    trace_count: int
    timestep_count: int

    @abstractmethod
    def compute_intermediate_variables(self, *args: NDArray[np.integer]) -> Dict[str, NDArray[np.integer]]:
        pass

    @abstractmethod
    def __getitem__(self, _idx: Union[int, Sequence[int]]) -> Tuple[NDArray[np.floating], NDArray[np.integer], Dict[str, NDArray[np.integer]]]:
        pass
    
    def __len__(self) -> int:
        assert hasattr(self, 'trace_count')
        return self.trace_count

    def __del__(self):
        database: Optional[h5py.File] = getattr(self, 'database', None)
        if database is not None:
            database.close()

class Base_TorchDataset(ABC, Dataset):
    def __init__(
            self,
            *,
            transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
            target_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    ):
        super().__init__()
        self.transform = transform
        self.target_transform = target_transform
    
    @abstractmethod
    def __getitem__(self, _idx: Union[int, Sequence[int]]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass