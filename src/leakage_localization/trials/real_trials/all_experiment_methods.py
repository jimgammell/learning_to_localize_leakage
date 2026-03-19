from typing import Optional, Dict, Any
from collections import defaultdict
import json
import pickle
from copy import copy
from tqdm import tqdm

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

def get_best_all_pretrain_hparams(sweep_dir: str):
    trial_indices = [int(x.split('_')[1]) for x in os.listdir(sweep_dir) if x.split('_')[0] == 'trial']
    trial_count = max(trial_indices) + 1
    assert all(os.path.exists(os.path.join(sweep_dir, f'trial_{x}')) for x in range(trial_count))
    best_val_rank = np.inf
    best_hparams = None
    for trial_idx in tqdm(range(trial_count)):
        dirpath = os.path.join(sweep_dir, f'trial_{trial_idx}')
        with open(os.path.join(dirpath, 'training_curves.pickle'), 'rb') as f:
            training_curves = pickle.load(f)
        val_rank = np.min(training_curves['val_theta_rank'][-1])
        if val_rank < best_val_rank:
            best_val_rank = val_rank
            with open(os.path.join(dirpath, 'hparams.pickle'), 'rb') as f:
                best_hparams = pickle.load(f)
    return best_hparams

def get_best_all_hparams(sweep_dir: str, oracle_assessment: np.ndarray, reference_dnn: nn.Module, reference_dataloader: Dataset):
    trial_indices = [int(x.split('_')[1]) for x in os.listdir(sweep_dir) if x.split('_')[0] == 'trial']
    trial_count = max(trial_indices) + 1
    assert all(os.path.exists(os.path.join(sweep_dir, f'trial_{x}')) for x in range(trial_count))
    mean_leakage_assessment = np.stack([
        np.load(os.path.join(sweep_dir, f'trial_{trial_idx}', 'leakage_assessment.npy')) for trial_idx in range(trial_count)
    ]).mean(axis=0)
    collected_results = defaultdict(list)
    for trial_idx in tqdm(range(trial_count)):
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
    
    fair_best_idx = np.argmin(collected_results['composite_criterion'])
    with open(os.path.join(sweep_dir, f'trial_{fair_best_idx}', 'hparams.json'), 'r') as f:
        fair_best_hparams = json.load(f)
    oracle_best_idx = np.argmax(collected_results['oracle_agreement'])
    with open(os.path.join(sweep_dir, f'trial_{oracle_best_idx}', 'hparams.json'), 'r') as f:
        oracle_best_hparams = json.load(f)
    return {
        'fair': fair_best_hparams,
        'oracle': oracle_best_hparams
    }

def train_all_model(
    output_dir: str,
    profiling_dataset: Dataset,
    attack_dataset: Dataset,
    training_kwargs: Optional[Dict[str, Any]] = None,
    max_steps: int = 1000,
    seed: int = 0,
    reference_leakage_assessment: Optional[np.ndarray] = None,
    pretrain_max_steps: int = 0,
    pretrain_kwargs: Optional[Dict[str, Any]] = None,
    pretrain_classifiers_dir: Optional[str] = None
):
    if os.path.exists(os.path.join(output_dir, 'leakage_assessment.npy')):
        return
    if training_kwargs is None:
        training_kwargs = {}
    trainer = ALLTrainer(
        profiling_dataset, attack_dataset, default_training_module_kwargs=training_kwargs, reference_leakage_assessment=reference_leakage_assessment
    )
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    if (pretrain_max_steps > 0) and (pretrain_classifiers_dir is None):
        if pretrain_kwargs is None:
            pretrain_kwargs = {}
        kwargs = copy(training_kwargs)
        kwargs.update(pretrain_kwargs)
        set_seed(seed) # important: we set the seed twice so both training phases will get the same train/val split. Stupid hacky solution but I'm lazy so it's what I'm doing.
        trainer.pretrain_classifiers(
            os.path.join(output_dir, 'classifier_pretraining'),
            max_steps=pretrain_max_steps,
            override_kwargs=kwargs
        )
        pretrain_classifiers_dir = os.path.join(output_dir, 'classifier_pretraining')
    set_seed(seed)
    trainer.run(
        os.path.join(output_dir, 'all_training'),
        pretrained_classifiers_logging_dir=pretrain_classifiers_dir,
        max_steps=max_steps,
        anim_gammas=False,
        reference=reference_leakage_assessment
    )
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    if not os.path.exists(os.path.join(output_dir, 'training_time.npy')):
        np.save(os.path.join(output_dir, 'training_time.npy'), elapsed_time)

def evaluate_all_hparam_sensitivity(
    output_dir: str,
    profiling_dataset: Dataset,
    attack_dataset: Dataset,
    training_kwargs: Optional[Dict[str, Any]] = None,
    max_steps: int = 1000,
    seed: int = 0,
    reference_leakage_assessment: Optional[np.ndarray] = None,
    pretrain_max_steps: int = 0,
    pretrain_kwargs: Optional[Dict[str, Any]] = None,
    pretrain_classifiers_dir: str = None
):
    gamma_bar_vals = np.arange(0.05, 1.0, 0.05)
    theta_lr_scalar_vals = np.logspace(-2, 2, 19)
    etat_lr_scalar_vals = np.logspace(-2, 2, 19)
    progress_bar = tqdm(total=len(gamma_bar_vals)+len(theta_lr_scalar_vals)+len(etat_lr_scalar_vals))
    for gamma_bar_val in gamma_bar_vals:
        trial_dir = os.path.join(output_dir, f'gamma_bar={gamma_bar_val}')
        if not os.path.exists(os.path.join(trial_dir, 'all_training', 'leakage_assessment.npy')):
            hparams = copy(training_kwargs)
            hparams.update({'gamma_bar': gamma_bar_val})
            train_all_model(
                trial_dir, profiling_dataset, attack_dataset, hparams, max_steps=max_steps, seed=seed, reference_leakage_assessment=reference_leakage_assessment,
                pretrain_max_steps=pretrain_max_steps, pretrain_kwargs=pretrain_kwargs, pretrain_classifiers_dir=pretrain_classifiers_dir
            )
        progress_bar.update(1)
    for theta_lr_scalar_val in theta_lr_scalar_vals:
        trial_dir = os.path.join(output_dir, f'theta_lr_scalar={theta_lr_scalar_val}')
        if os.path.exists(os.path.join(trial_dir, 'all_training', 'leakage_assessment.npy')):
            hparams = copy(training_kwargs)
            hparams['theta_lr'] *= theta_lr_scalar_val
            train_all_model(
                trial_dir, profiling_dataset, attack_dataset, hparams, max_steps=max_steps, seed=seed, reference_leakage_assessment=reference_leakage_assessment,
                pretrain_max_steps=pretrain_max_steps, pretrain_kwargs=pretrain_kwargs, pretrain_classifiers_dir=pretrain_classifiers_dir
            )
        progress_bar.update(1)
    for etat_lr_scalar_val in etat_lr_scalar_vals:
        trial_dir = os.path.join(output_dir, f'etat_lr_scalar={etat_lr_scalar_val}')
        if os.path.exists(os.path.join(trial_dir, 'all_training', 'leakage_assessment.npy')):
            hparams = copy(training_kwargs)
            hparams['etat_lr'] *= etat_lr_scalar_val
            train_all_model(
                trial_dir, profiling_dataset, attack_dataset, hparams, max_steps=max_steps, seed=seed, reference_leakage_assessment=reference_leakage_assessment,
                pretrain_max_steps=pretrain_max_steps, pretrain_kwargs=pretrain_kwargs, pretrain_classifiers_dir=pretrain_classifiers_dir
            )
        progress_bar.update(1)