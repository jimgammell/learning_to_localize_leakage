from typing import Optional, Dict, Any
from collections import defaultdict
import json

import numpy as np
from scipy.stats import spearmanr
import torch
from torch import nn
from torch.utils.data import Dataset

from common import *
from training_modules.adversarial_leakage_localization import ALLTrainer
from . import evaluation_methods

def run_all_hparam_sweep(
    output_dir: str,
    profiling_dataset: Dataset,
    attack_dataset: Dataset,
    training_kwargs: Optional[Dict[str, Any]] = None,
    classifiers_pretrain_trial_count: int = 0,
    trial_count: int = 50,
    max_classifiers_pretrain_steps: int = 0,
    max_steps: int = 1000,
    starting_seed: int = 0,
    reference_leakage_assessment: Optional[np.ndarray] = None
):
    if not os.path.exists(os.path.join(output_dir, 'results.pickle')):
        if training_kwargs is None:
            training_kwargs = {}
        trainer = ALLTrainer(
            profiling_dataset, attack_dataset, default_training_module_kwargs=training_kwargs, reference_leakage_assessment=reference_leakage_assessment
        )
        if classifiers_pretrain_trial_count > 0:
            assert max_classifiers_pretrain_steps > 0
            best_classifiers_pretrain_dir = trainer.htune_pretrain_classifiers(
                os.path.join(output_dir, 'classifiers_pretraining'),
                trial_count=classifiers_pretrain_trial_count,
                max_steps=max_classifiers_pretrain_steps,
                starting_seed=starting_seed
            )
        else:
            best_classifiers_pretrain_dir = None
        trainer.htune_leakage_localization(
            output_dir,
            pretrained_classifiers_logging_dir=best_classifiers_pretrain_dir,
            trial_count=trial_count,
            max_steps=max_steps,
            ablation='none',
            starting_seed=starting_seed+classifiers_pretrain_trial_count
        )

def get_best_all_hparams(sweep_dir: str, oracle_assessment: np.ndarray, reference_dnn: nn.Module, reference_dataloader: Dataset):
    trial_indices = [int(x.split('_')[1]) for x in os.listdir(sweep_dir) if x.split('_')[0] == 'trial']
    trial_count = max(trial_indices) + 1
    assert all(os.path.exists(os.path.join(sweep_dir, f'trial_{x}')) for x in range(trial_count))
    mean_leakage_assessment = np.stack([
        np.load(os.path.join(sweep_dir, f'trial_{trial_idx}', 'leakage_assessment.npy')) for trial_idx in range(trial_count)
    ]).mean(axis=0)
    collected_results = defaultdict(list)
    for trial_idx in range(trial_count):
        dirpath = os.path.join(sweep_dir, f'trial_{trial_idx}')
        if not os.path.exists(os.path.join(dirpath, 'metrics.npz')):
            leakage_assessment = np.load(os.path.join(dirpath, 'leakage_assessment.npy'))
            metrics = {}
            metrics['oracle_agreement'] = evaluation_methods.get_oracle_agreement(leakage_assessment, oracle_assessment)
            metrics['fwd_dnno_criterion'] = evaluation_methods.get_forward_dnno_criterion(leakage_assessment, reference_dnn, reference_dataloader)
            metrics['rev_dnno_criterion'] = evaluation_methods.get_reverse_dnno_criterion(leakage_assessment, reference_dnn, reference_dataloader)
            if leakage_assessment.std() > 0:
                metrics['mean_agreement'] = spearmanr(leakage_assessment.reshape(-1), mean_leakage_assessment.reshape(-1)).statistic
            else:
                metrics['mean_agreement'] = 0.
            np.savez(os.path.join(dirpath, 'metrics.npz'), **metrics)
        else:
            metrics = {key: val for key,val in np.load(os.path.join(dirpath, 'metrics.npz'), allow_pickle=True).items()}
        for key, val in metrics.items():
            collected_results[key].append(val)
    collected_results = {key: np.array(val) for key, val in collected_results.items()}
    collected_results['composite_criterion'] = (
        collected_results['fwd_dnno_criterion'].argsort().argsort()
        + (-collected_results['rev_dnno_criterion']).argsort().argsort()
        + (-collected_results['mean_agreement']).argsort().argsort()
    )
    collected_results['composite_dnno_criterion'] = (
        collected_results['fwd_dnno_criterion'].argsort().argsort()
        + (-collected_results['rev_dnno_criterion']).argsort().argsort()
    )
    fig, axes = plt.subplots(1, 5, figsize=(5*PLOT_WIDTH, PLOT_WIDTH))
    for ax, key in zip(axes, ['fwd_dnno_criterion', 'rev_dnno_criterion', 'mean_agreement', 'composite_criterion', 'composite_dnno_criterion']):
        ax.plot(collected_results['oracle_agreement'], collected_results[key], marker='.', linestyle='none')
        ax.set_xlabel('Oracle agreement')
        ax.set_ylabel(key)
    fig.tight_layout()
    fig.savefig(os.path.join(sweep_dir, 'selection_criterion.png'))
    plt.close(fig)
    
    best_idx = np.argmin(collected_results['composite_dnno_criterion'])
    with open(os.path.join(sweep_dir, f'trial_{best_idx}', 'hparams.json'), 'r') as f:
        best_hparams = json.load(f)
    return best_hparams

def train_all_model(
    output_dir: str,
    profiling_dataset: Dataset,
    attack_dataset: Dataset,
    training_kwargs: Optional[Dict[str, Any]] = None,
    max_steps: int = 1000,
    seed: int = 0,
    reference_leakage_assessment: Optional[np.ndarray] = None,
    pretrain_classifiers: bool = False
):
    if training_kwargs is None:
        training_kwargs = {}
    trainer = ALLTrainer(
        profiling_dataset, attack_dataset, default_training_module_kwargs=training_kwargs, reference_leakage_assessment=reference_leakage_assessment
    )
    set_seed(seed)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    if pretrain_classifiers:
        pass
    assert False