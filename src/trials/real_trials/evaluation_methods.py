from typing import Literal, Optional
from tqdm import tqdm

import numpy as np
from scipy.stats import spearmanr
import torch
from torch.utils.data import TensorDataset, DataLoader

from utils.metrics import get_rank
from utils.aes_multi_trace_eval import AESMultiTraceEvaluator
from utils.template_attack import TemplateAttack

def get_oracle_agreement(leakage_assessment, oracle_assessment):
    assert leakage_assessment.shape[-1] == oracle_assessment.shape[-1]
    if leakage_assessment.std() == 0:
        agreement = 0.
    else:
        agreement = spearmanr(leakage_assessment.reshape(-1), oracle_assessment.reshape(-1)).statistic
    return agreement

@torch.no_grad()
def run_dnn_occlusion_test(
    leakage_assessment, model, traces, labels,
    dataloader: Optional[DataLoader] = None,
    performance_metric: Literal['mean_rank', 'traces_to_disclosure'] = 'mean_rank',
    direction: Literal['forward', 'reverse'] = 'reverse',
    dataset_name: Optional[Literal['dpav4', 'aes-hd', 'ascasdv1-fixed', 'ascadv1-variable']] = None,
    parallel_pass_count: int = 25,
    quiet: bool = True
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    traces = traces.to(device)
    labels = labels.to(device)
    timesteps_per_trace = leakage_assessment.shape[-1]
    leakage_ranking = leakage_assessment.reshape(-1).argsort()
    if direction == 'forward':
        leakage_ranking = leakage_ranking[::-1].copy()
    else:
        assert direction == 'reverse'
    mask = torch.zeros(1, timesteps_per_trace, dtype=torch.float, device=device)
    performance = []
    idx = 0
    batch_size, *trace_dims = traces.shape
    if not quiet:
        progress_bar = tqdm(total=len(leakage_ranking))
    while idx < len(leakage_ranking):
        masked_trace_batch = []
        for _ in range(parallel_pass_count):
            if idx < len(leakage_ranking):
                mask[:, leakage_ranking[idx]] = 1.
                masked_traces = mask.unsqueeze(0)*traces
                masked_trace_batch.append(masked_traces)
            idx += 1
        _parallel_pass_count = len(masked_trace_batch)
        masked_trace_batch = torch.stack(masked_trace_batch).reshape(batch_size*_parallel_pass_count, *trace_dims)
        logits_batch = model(masked_trace_batch).reshape(_parallel_pass_count, batch_size, -1).cpu()
        if performance_metric == 'mean_rank':
            for logits in logits_batch:
                rank = get_rank(logits, labels).mean()
                performance.append(rank)
        elif performance_metric == 'traces_to_disclosure':
            assert dataset_name is not None
            for logits in logits_batch:
                evaluator = AESMultiTraceEvaluator(dataloader, model, seed=0, device=device, dataset_name=dataset_name)
                rank_over_time = evaluator(logits=logits)
                traces_to_disclosure = np.nonzero(rank_over_time-1)[0][-1] + 1 if len(np.nonzero(rank_over_time-1)) > 0 else 1
                performance.append(traces_to_disclosure)
        else:
            assert False
        if not quiet:
            progress_bar.update(_parallel_pass_count)
    return np.array(performance)

def get_forward_dnno_criterion(
        leakage_assessment, model, dataloader,
        reduce: bool = True,
        pass_through_dataloader: bool = False,
        **kwargs
    ):
    batches = list(dataloader)
    traces = torch.cat([x[0] for x in batches], dim=0)
    labels = torch.cat([x[1] for x in batches], dim=0)
    if pass_through_dataloader:
        kwargs['dataloader'] = dataloader
    rv = run_dnn_occlusion_test(
        leakage_assessment, model, traces, labels, direction='forward', **kwargs
    )
    if reduce:
        rv = rv.mean()
    return rv

def get_reverse_dnno_criterion(
        leakage_assessment, model, dataloader,
        reduce: bool = True,
        pass_through_dataloader: bool = False,
        **kwargs
    ):
    batches = list(dataloader)
    traces = torch.cat([x[0] for x in batches], dim=0)
    labels = torch.cat([x[1] for x in batches], dim=0)
    if pass_through_dataloader:
        kwargs['dataloader'] = dataloader
    rv = run_dnn_occlusion_test(
        leakage_assessment, model, traces, labels, direction='reverse', **kwargs
    )
    if reduce:
        rv = rv.mean()
    return rv

@torch.no_grad()
def run_template_attack_test(
    leakage_assessment, profiling_dataset, attack_dataset, topk=10, dataset_name=None
):
    if dataset_name in ['ascadv1-fixed', 'ascadv1-variable', 'aes-hd', 'dpav4']: # AES datasets where we accumulate prediction over multiple traces
        if dataset_name in ['ascadv1-fixed', 'ascadv1-variable']:
            from datasets.ascadv1 import to_key_preds
            arg_keys = ['plaintext']
            constants = None
        elif dataset_name == 'dpav4':
            from datasets.dpav4 import to_key_preds
            arg_keys = ['plaintext', 'offset']
            constants = [profiling_dataset.mask]
        elif dataset_name == 'aes-hd':
            from datasets.aes_hd import to_key_preds
            arg_keys = ['ciphertext_11', 'ciphertext_7']
            constants = None
        if dataset_name in ['ascadv1-fixed', 'ascadv1-variable']: # second-order datasets, so we want to increase likelihood of catching all the leaky variables
            leakage_assessment = leakage_assessment.reshape(10, -1)
            pois = leakage_assessment.argsort(axis=-1) + leakage_assessment.shape[-1]*np.arange(10).reshape(-1, 1)
            pois = pois[:, -2:]
            pois = pois.reshape(-1)
        else:
            pois = leakage_assessment.argsort()[-20:]
        template_attacker = TemplateAttack(pois, 'label')
        template_attacker.profile(profiling_dataset)
        rv = template_attacker.attack(attack_dataset, arg_keys=arg_keys, constants=constants, int_var_to_key_fn=to_key_preds)
        ttd = []
        for _rv_seed in rv:
            traces_to_disclosure = np.nonzero(_rv_seed-1)[0][-1] + 1 if len(np.nonzero(_rv_seed-1)[0]) > 0 else 1
            ttd.append(traces_to_disclosure)
        ttd = np.array(ttd).mean()
        return ttd
    elif dataset_name in ['otiait', 'otp']:
        pois = leakage_assessment.argsort()[-20:]
        template_attacker = TemplateAttack(pois, 'label')
        template_attacker.profile(profiling_dataset)
        ranks = template_attacker.get_ranks(attack_dataset).mean()
        return ranks