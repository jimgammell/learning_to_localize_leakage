from typing import Literal, Optional

import numpy as np
from scipy.stats import spearmanr
import torch
from torch.utils.data import TensorDataset, DataLoader

from utils.metrics import get_rank
from utils.aes_multi_trace_eval import AESMultiTraceEvaluator

def get_oracle_agreement(leakage_assessment, oracle_assessment):
    assert leakage_assessment.shape[-1] == oracle_assessment.shape[-1]
    if leakage_assessment.std() == 0:
        agreement = 0.
    else:
        agreement = spearmanr(leakage_assessment.reshape(-1), oracle_assessment.reshape(-1)).statistic
    return agreement

def run_dnn_occlusion_test(
    leakage_assessment, model, traces, labels,
    performance_metric: Literal['mean_rank', 'traces_to_disclosure'] = 'mean_rank',
    direction: Literal['forward', 'reverse'] = 'reverse',
    dataset_name: Optional[Literal['dpav4', 'aes-hd', 'ascasdv1-fixed', 'ascadv1-variable']] = None
):
    device = list(model.parameters())[0].device
    timesteps_per_trace = leakage_assessment.shape[-1]
    leakage_ranking = leakage_assessment.reshape(-1).argsort()
    if direction == 'forward':
        leakage_ranking = leakage_ranking[::-1].copy()
    else:
        assert direction == 'reverse'
    mask = torch.zeros(1, timesteps_per_trace, dtype=torch.float, device=device)
    performance = []
    for idx in leakage_ranking:
        mask[:, idx] = 1.
        masked_traces = mask.unsqueeze(0)*traces
        if performance_metric == 'mean_rank':
            logits = model(masked_traces)
            rank = get_rank(logits, labels)
            performance.append(rank)
        elif performance_metric == 'traces_to_disclosure':
            assert dataset_name is not None
            dataset = TensorDataset(masked_traces, labels)
            dataloader = DataLoader(dataset, batch_size=len(traces))
            evaluator = AESMultiTraceEvaluator(dataloader, model, seed=0, device=device, dataset_name=dataset_name)
            rank_over_time = evaluator()
            traces_to_disclosure = np.nonzero(rank_over_time-1)[0][-1] + 1 if len(np.nonzero(rank_over_time-1)) > 0 else 1
            performance.append(traces_to_disclosure)
        else:
            assert False
    return np.array(performance)

def get_forward_dnno_criterion(leakage_assessment, model, dataloader):
    batches = list(dataloader)
    traces = torch.cat([x[0] for x in batches], dim=0)
    labels = torch.cat([x[1] for x in batches], dim=0)
    return run_dnn_occlusion_test(
        leakage_assessment, model, traces, labels, performance_metric='mean_rank', direction='forward'
    ).mean()

def get_reverse_dnno_criterion(leakage_assessment, model, dataloader):
    batches = list(dataloader)
    traces = torch.cat([x[0] for x in batches], dim=0)
    labels = torch.cat([x[1] for x in batches], dim=0)
    return run_dnn_occlusion_test(
        leakage_assessment, model, traces, labels, performance_metric='mean_rank', direction='reverse'
    ).mean()