from typing import Optional, Callable, Sequence
from math import sqrt

import numpy as np
import torch
from torch.utils.data import Dataset

class MultiXORDataset(Dataset):
    def __init__(self,
        buffer_size: int = 10000,
        second_order_pair_count: int = 0,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.random_feature_count = 1
        self.second_order_pair_count = second_order_pair_count
        self.transform = transform
        self.target_transform = target_transform
        self.timesteps_per_trace = self.random_feature_count + 2*self.second_order_pair_count
        self.class_count = 2
        self.return_metadata = False
        self.item_iterator = None
    
    def get_item_iterator(self):
        while True:
            datapoints, labels = self.generate_data()
            for datapoint, label in zip(datapoints, labels):
                if self.transform is not None:
                    datapoint = self.transform(datapoint)
                if self.target_transform is not None:
                    label = self.target_transform(label)
                yield datapoint, label
    
    def generate_data(self):
        labels = np.random.randint(2, size=(self.buffer_size,), dtype=np.int64)
        random_features = (
            np.random.randn(self.buffer_size, 1, self.random_feature_count).astype(np.float32)
            + (2*np.random.randint(2, size=(self.buffer_size, 1, self.random_feature_count)).astype(np.float32)-1)
        )
        if self.random_feature_count > 0:
            masks = np.random.randint(2, size=(self.buffer_size, self.second_order_pair_count), dtype=np.int64)
            masked_labels = labels.reshape(self.buffer_size, 1) ^ masks
            mask_features = (
                np.random.randn(self.buffer_size, 1, self.second_order_pair_count).astype(np.float32)
                + (2*masks.reshape(self.buffer_size, 1, self.second_order_pair_count).astype(np.float32)-1)
            )
            masked_label_features = (
                np.random.randn(self.buffer_size, 1, self.second_order_pair_count).astype(np.float32)
                + (2*masked_labels.reshape(self.buffer_size, 1, self.second_order_pair_count).astype(np.float32)-1)
            )
            datapoints = np.concatenate([random_features, mask_features, masked_label_features], axis=2)
            return datapoints, labels
        else:
            return random_features, labels
    
    def __getitem__(self, idx):
        if self.item_iterator is None:
            self.item_iterator = self.get_item_iterator()
        if isinstance(idx, Sequence):
            datapoints, labels = [], []
            for _ in idx:
                datapoint, label = next(self.item_iterator)
                datapoints.append(datapoint)
                labels.append(label)
            datapoints = np.stack(datapoints)
            labels = np.stack(labels)
        else:
            datapoints, labels = next(self.item_iterator)
        if self.return_metadata:
            return datapoints, labels, {'label': labels}
        else:
            return datapoints, labels
    
    def __len__(self):
        return self.buffer_size