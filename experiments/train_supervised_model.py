from typing import Dict, Any, Tuple, List, get_args, Optional
from copy import copy
import argparse
import fcntl
import logging
import subprocess
from functools import partial

import yaml
import pandas
import numpy as np
from torch.utils.data import Dataset, Subset
import lightning
import optuna
from leakage_localization.datasets import Base_TorchDataset
from leakage_localization.training import SupervisedModule
from leakage_localization.training.train_supervised_model import train_supervised_model
from leakage_localization.training.hyperparameter_tuning import SamplerType, PruningCallback, sample_hparams, get_study, generate_qmc_trials
from leakage_localization.models import Model

from init_things import *
from utils.load_things import load_torch_dataset, construct_loaders
from utils.training_config import SupervisedTrainingConfig

# function by Claude to override particular config arguments from the command line
def _apply_overrides(config: Dict[str, Any], overrides: list) -> None:
    i = 0
    while i < len(overrides):
        arg = overrides[i]
        if not arg.startswith('--'):
            raise ValueError(f'Expected --key.subkey value, got: {arg}')
        key = arg[2:]
        if i + 1 >= len(overrides) or overrides[i + 1].startswith('--'):
            raise ValueError(f'No value provided for override {arg}')
        raw_value = overrides[i + 1]
        i += 2
        keys = key.split('.')
        d = config
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                raise ValueError(f'Invalid config path: {key}')
            d = d[k]
        if keys[-1] not in d:
            raise ValueError(f'Key {key} not found in config')
        d[keys[-1]] = yaml.safe_load(raw_value)

def construct_datasets(config: SupervisedTrainingConfig) -> Tuple[Dataset, ...]:
    profiling_set = load_torch_dataset(
        config.data.id,
        'profile',
        target_byte=config.data.target_byte,
        target_variable=config.data.target_variable
    )
    attack_set = load_torch_dataset(
        config.data.id,
        'attack',
        target_byte=config.data.target_byte,
        target_variable=config.data.target_variable
    )
    indices = np.random.default_rng(seed=str_to_seed('data_partition')).permutation(len(profiling_set))
    val_len = int(len(indices)*config.data.val_prop)
    train_set = Subset(copy(profiling_set), indices=indices[:-val_len])
    val_set = Subset(copy(profiling_set), indices=indices[-val_len:])
    test_set = copy(attack_set)
    return profiling_set, attack_set, train_set, val_set, test_set

def construct_module(profiling_set: Base_TorchDataset, config: SupervisedTrainingConfig) -> SupervisedModule:
    module = SupervisedModule(
        model_constructor=Model,
        model_kwargs=dict(
            input_length=profiling_set.timestep_count,
            output_count=(
                66 if config.model.grey_box_head == 'ascadv1' else
                18 if config.model.grey_box_head == 'ascadv2' else
                profiling_set.config.num_labels
            ),
            grey_box_head=config.model.grey_box_head,
            trunk=config.model.trunk,
            position_embedding=config.model.position_embedding,
            pooling=config.model.pooling,
            head=config.model.head,
            fnn_style=config.model.fnn_style,
            patch_size=config.model.patch_size,
            use_fourier_embed=config.model.use_fourier_embed,
            fourier_embed_num_bands=config.model.fourier_embed_num_bands,
            fourier_embed_sigma=config.model.fourier_embed_sigma,
            embedding_dim=config.model.embedding_dim,
            expansion_factor=config.model.expansion_factor,
            trunk_blocks=config.model.trunk_blocks,
            head_count=config.model.head_count,
            register_tokens=config.model.register_tokens,
            input_dropout_rate=config.model.input_dropout_rate,
            input_droppatch_rate=config.model.input_droppatch_rate,
            hidden_dropout_rate=config.model.hidden_dropout_rate,
            use_bias=config.model.use_bias,
            perceiver_latent_dim=config.model.perceiver_latent_dim,
            perceiver_self_attn_per_cross_attn_blocks=config.model.perceiver_self_attn_per_cross_attn_blocks,
            perceiver_cross_attn_head_count=config.model.perceiver_cross_attn_head_count
        ),
        leakage_model=config.model.leakage_model,
        num_labels=profiling_set.config.num_labels,
        total_steps=config.training.total_steps,
        lr_warmup_steps=int(config.training.total_steps*config.training.lr_warmup_frac),
        lr_const_steps=int(config.training.total_steps*config.training.lr_const_frac),
        base_lr=config.training.base_lr,
        lr_decay_multiplier=config.training.lr_decay_multiplier,
        weight_decay=config.training.weight_decay,
        label_smoothing=config.training.label_smoothing,
        mtd_kwargs={
            'target_preds_to_key_preds': profiling_set.target_preds_to_key_preds,
            'int_var_keys': profiling_set.int_var_keys,
            'attack_count': config.mtd.attack_count,
            'traces_per_attack': config.mtd.traces_per_attack
        },
        trace_statistics=profiling_set.get_trace_statistics(),
        additive_gaussian_noise=config.training.additive_gaussian_noise,
        mixup_alpha=config.training.mixup_alpha,
        preprocessing=config.data.preprocessing,
        random_roll_scale=config.data.random_roll_scale,
        random_lpf_scale=config.data.random_lpf_scale,
    )
    if config.training.compile:
        module.model.compile()
    return module

def run_train_model(
        dest: Path,
        config: SupervisedTrainingConfig,
        aux_callbacks: Optional[List[lightning.Callback]] = None
):
    set_seed(config.training.seed)
    profiling_set, attack_set, train_set, val_set, test_set = construct_datasets(config)
    train_loader, val_loader, test_loader = construct_loaders([train_set], [val_set, test_set], batch_size=config.training.batch_size, num_workers=config.training.num_workers)
    training_module = construct_module(profiling_set, config)
    train_supervised_model(
        dest=dest,
        training_module=training_module,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        total_steps=config.training.total_steps,
        grad_clip_val=config.training.grad_clip_val,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        early_stop_metric=config.training.early_stop_metric,
        early_stop_mode=config.training.early_stop_mode,
        aux_callbacks=aux_callbacks
    )

def _optuna_objective(
        trial: optuna.Trial,
        dest: Path,
        config: SupervisedTrainingConfig,
        enable_pruning: bool = False,
        use_trial_subdir: bool = True
) -> float:
    trial_dest = dest / f'trial_{trial.number}' if use_trial_subdir else dest
    config.training.seed = trial.number
    for field_key, field_search_space in config.search_space.items():
        new_hparams = sample_hparams(trial, field_search_space)
        assert hasattr(config, field_key)
        for k, v in new_hparams.items():
            assert hasattr(getattr(config, field_key), k)
            setattr(getattr(config, field_key), k, v)
    aux_callbacks = []
    if enable_pruning:
        aux_callbacks.append(PruningCallback(trial, config.training.early_stop_metric))
    run_train_model(trial_dest, config, aux_callbacks=aux_callbacks if aux_callbacks else None)
    metrics = pandas.read_csv(trial_dest / 'metrics.csv')
    tracked_metric = metrics[config.training.early_stop_metric]
    if config.training.early_stop_mode == 'max':
        objective = tracked_metric.max()
    elif config.training.early_stop_mode == 'min':
        objective = tracked_metric.min()
    else:
        assert False
    return objective

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', required=True, type=str)
    parser.add_argument('--dest', required=True, type=Path)
    parser.add_argument('--config-root', type=Path, default=CONFIG_ROOT)
    parser.add_argument('--optuna-study-path', type=Path, default=None)
    parser.add_argument('--optuna-run-count', type=int, default=1)
    parser.add_argument('--optuna-enable-pruning', default=False, action='store_true')
    parser.add_argument('--optuna-sampler', default='random', choices=get_args(SamplerType))
    parser.add_argument('--optuna-total-trials', type=int, default=None,
                        help='When using --optuna-sampler qmc, pre-enqueue this many QMC trials '
                             'before calling study.optimize(). Each parallel worker acquires a '
                             'file lock so the pre-generation runs in exactly one process.')
    append_directory_clargs(parser)
    args, overrides = parser.parse_known_args()
    init_directories(vars(args), load_directory_config())

    dest: Path = args.dest
    assert isinstance(dest, Path)
    config_root: Path = args.config_root
    assert isinstance(config_root, Path) and config_root.exists()
    config_path = config_root / f'{args.config_file}.yaml'
    assert config_path.exists()
    dest.mkdir(exist_ok=True, parents=True)
    optuna_study_path: Optional[Path] = args.optuna_study_path
    optuna_run_count: int = args.optuna_run_count
    optuna_enable_pruning: bool = args.optuna_enable_pruning
    optuna_sampler_type: SamplerType = args.optuna_sampler
    optuna_total_trials: Optional[int] = args.optuna_total_trials
    if optuna_study_path is not None:
        assert len(overrides) == 0
        assert isinstance(optuna_study_path, Path)
        assert isinstance(optuna_run_count, int) and optuna_run_count >= 1
        assert isinstance(optuna_enable_pruning, bool)
        assert optuna_sampler_type in get_args(SamplerType)
    with open(config_path, 'r') as f:
        config_kw = safe_load_yaml(f)
    _apply_overrides(config_kw, overrides)

    config_kw['commit_hash'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    config_path = dest / 'config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            existing_config = yaml.safe_load(f)
        existing_hash = existing_config.pop('commit_hash', None)
        current_hash = config_kw.pop('commit_hash', None)
        if existing_hash != current_hash:
            logging.warning(f'Resuming trial with a different commit hash. Current hash: {current_hash}. Existing hash: {existing_hash}.')
        assert config_kw == existing_config
        config_kw['commit_hash'] = current_hash
    else:
        with open(config_path, 'w') as f:
            yaml.dump(config_kw, f)
    config = SupervisedTrainingConfig(**config_kw)

    if optuna_study_path is not None:
        optuna_study_path.parent.mkdir(exist_ok=True, parents=True)
        optuna_study = get_study(
            optuna_study_path,
            study_direction={'min': 'minimize', 'max': 'maximize'}[config.training.early_stop_mode],
            sampler_type=optuna_sampler_type,
            enable_pruning=optuna_enable_pruning,
            seed=SEED
        )
        if optuna_sampler_type == 'qmc' and optuna_total_trials is not None:
            # Enqueue all QMC hyperparameter configurations before any worker starts
            # sampling, avoiding the race condition in QMCSampler._find_sample_id.
            # A file lock ensures only one worker runs the (idempotent) enqueue step.
            lock_path = optuna_study_path.with_suffix('.pregen.lock')
            with open(lock_path, 'w') as _lock_file:
                fcntl.flock(_lock_file, fcntl.LOCK_EX)
                generate_qmc_trials(
                    study=optuna_study,
                    search_space=config.search_space,
                    n_trials=optuna_total_trials,
                    seed=SEED,
                )
        optuna_objective = partial(_optuna_objective, dest=dest, config=config, enable_pruning=optuna_enable_pruning, use_trial_subdir=(optuna_run_count > 1))
        n_complete = sum(1 for t in optuna_study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        if optuna_total_trials is not None and n_complete >= optuna_total_trials:
            logging.info(f'Study already has {n_complete} complete trials (>= {optuna_total_trials}). Skipping training.')
        else:
            optuna_study.optimize(optuna_objective, n_trials=optuna_run_count)
    else:
        run_train_model(dest, config)

if __name__ == '__main__':
    main()