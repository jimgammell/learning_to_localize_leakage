from typing import *
import os
from copy import copy
import numpy as np
from scipy.special import log_softmax
import torch
from torch import nn
from torch.utils.data import Subset, DataLoader

from utils.metrics.rank import get_rank

class ReshapeOutput(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        logits = self.model(x)
        logits = logits.reshape(-1, logits.size(-1))
        return logits

class AESMultiTraceEvaluator:
    def __init__(self, dataloader: DataLoader, model: Union[nn.Module, str, os.PathLike], seed: Optional[int] = None, device: Optional[str] = None, dataset_name: Literal['dpav4', 'aes-hd', 'ascadv1-fixed', 'ascadv1-variable'] = 'dpav4'):
        self.dataloader = dataloader
        base_dataset = self.dataloader.dataset
        while isinstance(base_dataset, Subset):
            base_dataset = base_dataset.dataset
        self.base_dataset = base_dataset
        self.base_dataset.return_metadata = True
        self.base_model = model
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.base_model.eval()
        self.base_model.requires_grad_(False)
        self.base_model = self.base_model.to(self.device)
        self.trace_shape = self.base_model.input_shape
        self.model = ReshapeOutput(self.base_model)
        self.dataset_name = dataset_name
        if self.dataset_name == 'dpav4':
            from datasets.dpav4 import to_key_preds
        elif self.dataset_name == 'aes-hd':
            from datasets.aes_hd import to_key_preds
        elif self.dataset_name in ['ascadv1-fixed', 'ascadv1-variable']:
            from datasets.ascadv1 import to_key_preds
        else:
            to_key_preds = None
        self.to_key_preds = to_key_preds
    
    @torch.no_grad()
    def get_loss_and_rank(self):
        losses, ranks = [], []
        for batch_idx, (trace, label, _) in enumerate(self.dataloader):
            trace = trace.to(self.device)
            logits = self.model(trace).cpu()
            loss = nn.functional.cross_entropy(logits, label)
            rank = get_rank(logits, label).mean()
            losses.append(loss.item())
            ranks.append(rank.item())
        return np.mean(losses), np.mean(ranks)

    @torch.no_grad()
    def get_key_predictions(self):
        key_logitss = np.full((len(self.dataloader.dataset), 256), np.nan, dtype=np.float32)
        ground_truth_keys = np.full((len(self.dataloader.dataset),), -1, dtype=np.int64)
        losses = []
        for batch_idx, (trace, label, metadata) in enumerate(self.dataloader):
            trace = trace.to(self.device)
            batch_size = trace.shape[0]
            logits = self.model(trace).cpu()
            loss = nn.functional.cross_entropy(logits, label)
            losses.append(loss.detach().cpu().item())
            logits = logits.numpy()
            if self.dataset_name == 'dpav4':
                plaintexts = metadata['plaintext'].numpy()
                keys = metadata['key'].numpy()
                offsets = metadata['offset'].numpy()
                mask = self.base_dataset.mask
                for idx in range(batch_size):
                    save_idx = batch_idx*self.dataloader.batch_size + idx
                    key_logits = self.to_key_preds(logits[idx, :], np.stack([plaintexts[idx], offsets[idx]]), np.stack([mask]))
                    key_logitss[save_idx, :] = key_logits
                    ground_truth_keys[save_idx] = keys[idx]
            elif self.dataset_name == 'aes-hd':
                ciphertext_11 = metadata['ciphertext_11'].numpy()
                ciphertext_7 = metadata['ciphertext_7'].numpy()
                keys = metadata['key'].numpy()
                for idx in range(batch_size):
                    save_idx = batch_idx*self.dataloader.batch_size + idx
                    key_logits = self.to_key_preds(logits[idx, :], np.stack([ciphertext_11[idx], ciphertext_7[idx]]))
                    key_logitss[save_idx, :] = key_logits
                    ground_truth_keys[save_idx] = keys[idx]
            elif self.dataset_name in ['ascadv1-fixed', 'ascadv1-variable']:
                plaintexts = metadata['plaintext'].numpy()
                keys = metadata['key'].numpy()
                for idx in range(batch_size):
                    save_idx = batch_idx*self.dataloader.batch_size + idx
                    key_logits = self.to_key_preds(logits[idx, :], plaintexts[idx])
                    key_logitss[save_idx, :] = key_logits
                    ground_truth_keys[save_idx] = keys[idx]
            else:
                raise NotImplementedError
        assert np.all(np.isfinite(key_logitss))
        assert np.all(ground_truth_keys >= 0)
        return key_logitss, ground_truth_keys

    def get_rank_over_time(self, logitss: np.ndarray, ground_truth_keys: np.ndarray):
        accumulated_predictions = np.full((len(logitss), self.base_dataset.class_count), np.nan, dtype=np.float32)
        rank_over_time = np.full((len(logitss),), -1, dtype=np.int64)
        for idx, (logits, key) in enumerate(zip(logitss, ground_truth_keys)):
            if idx == 0:
                accumulated_predictions[idx, :] = log_softmax(logits)
            elif idx > 0:
                accumulated_predictions[idx, :] = log_softmax(logits) + accumulated_predictions[idx-1]
            else:
                assert False
            rank_over_time[idx] = get_rank(np.stack([accumulated_predictions[idx, :]]), np.stack([key]))[0]
        assert np.all(np.isfinite(accumulated_predictions)) and np.all(rank_over_time > -1)
        return rank_over_time
    
    def __call__(self, get_rank_over_time: bool = True):
        if get_rank_over_time:
            logitss, ground_truth_keys = self.get_key_predictions()
            rank_over_time = self.get_rank_over_time(logitss, ground_truth_keys)
            return rank_over_time
        else:
            loss, rank = self.get_loss_and_rank()
            return loss, rank