from pathlib import Path
from typing import Dict, Any, Tuple
from copy import copy
from math import ceil

from torch.utils.data import Dataset, Subset, DataLoader

from .entrypoint import main
from experiments.initialization import *
from leakage_localization.datasets.ascadv1 import ASCADv1_TorchDataset
from leakage_localization.datasets.transforms import Compose, Standardize, Normalize, RandomRoll, AdditiveGaussianNoise
from leakage_localization.training.train_supervised_model import train_supervised_model
from leakage_localization.training.supervised_lightning_module import SupervisedModule
from leakage_localization.models.model import Model

def construct_datasets(
        config: Dict[str, Any]
) -> Tuple[Dataset, Dataset, Dataset]:
    if config['data']['id'] == 'ascadv1-fixed':
        profiling_set = ASCADv1_TorchDataset(
            root=ASCADV1_FIXED_ROOT,
            partition='profile',
            target_byte=config['data']['target_byte'],
            target_variable=config['data']['target_variable'],
            variable_key=False,
            cropped_traces=False,
            binary_trace_file=True
        )
        attack_set = ASCADv1_TorchDataset(
            root=ASCADV1_FIXED_ROOT,
            partition='profile',
            target_byte=config['data']['target_byte'],
            target_variable=config['data']['target_variable'],
            variable_key=False,
            cropped_traces=False,
            binary_trace_file=True
        )
    else:
        assert False
    
    train_transforms, eval_transforms = [], []
    trace_statistics = profiling_set.get_trace_statistics(use_progress_bar=True)
    if config['data']['preprocessing'] == 'standardize':
        preprocessing_transform = Standardize(mean=trace_statistics['mean'], scale=np.sqrt(trace_statistics['var']))
    elif config['data']['preprocessing'] == 'normalize':
        preprocessing_transform = Normalize(min=trace_statistics['min'], max=trace_statistics['max'])
    else:
        assert False
    train_transforms.append(preprocessing_transform)
    eval_transforms.append(preprocessing_transform)
    if config['data']['additive_gaussian_noise'] > 0:
        train_transforms.append(AdditiveGaussianNoise(scale=config['data']['additive_gaussian_noise']))
    if config['data']['random_roll'] > 0:
        train_transforms.append(RandomRoll(max_shift=config['data']['random_roll']))
    train_transform = Compose(train_transforms)
    eval_transform = Compose(eval_transforms)
    train_set = copy(profiling_set)
    train_set.transform = train_transform
    val_set = copy(profiling_set)
    val_set.transform = eval_transform
    indices = np.random.default_rng(seed=0).permutation(len(profiling_set))
    val_len = int(len(indices)*config['data']['val_prop'])
    train_set = Subset(train_set, indices=indices[:-val_len])
    val_set = Subset(val_set, indices=indices[-val_len:])
    test_set = attack_set
    test_set.transform = eval_transform
    return train_set, val_set, test_set

def construct_training_module(
        train_set: Subset, val_set: Subset, test_set: Dataset,
        config: Dict[str, Any]
) -> SupervisedModule:
    module = SupervisedModule(
        model_constructor=Model,
        model_kwargs={
            'input_length': train_set.dataset.timestep_count,
            'output_dim': train_set.dataset.config.num_classes,
            'output_count': train_set.dataset.config.num_labels,
            'trunk': config['model']['trunk'],
            'position_embedding': config['model']['position_embedding'],
            'pooling': config['model']['pooling'],
            'head': config['model']['head'],
            'patch_size': config['model']['patch_size'],
            'use_fourier_embed': config['model']['use_fourier_embed'],
            'fourier_embed_num_bands': config['model']['fourier_embed_num_bands'],
            'fourier_embed_sigma': config['model']['fourier_embed_sigma'],
            'embedding_dim': config['model']['embedding_dim'],
            'expansion_factor': config['model']['expansion_factor'],
            'trunk_blocks': config['model']['trunk_blocks'],
            'head_count': config['model']['head_count'],
            'register_tokens': config['model']['register_tokens'],
            'input_dropout_rate': config['model']['input_dropout_rate'],
            'input_droppatch_rate': config['model']['input_droppatch_rate'],
            'hidden_dropout_rate': config['model']['hidden_dropout_rate'],
            'use_bias': config['model']['use_bias'],
            'perceiver_latent_dim': config['model']['perceiver_latent_dim'],
            'perceiver_self_attn_per_cross_attn_blocks': config['model']['perceiver_self_attn_per_cross_attn_blocks']
        },
        num_labels=train_set.dataset.config.num_labels,
        num_classes=train_set.dataset.config.num_classes,
        total_steps=config['training']['total_steps'],
        lr_warmup_steps=config['training']['lr_warmup_steps'],
        lr_const_steps=config['training']['lr_const_steps'],
        base_lr=config['training']['base_lr'],
        lr_decay_multiplier=config['training']['lr_decay_multiplier'],
        weight_decay=config['training']['weight_decay'],
        label_smoothing=config['training']['label_smoothing'],
        mttd_kwargs={
            'target_preds_to_key_preds': train_set.dataset.target_preds_to_key_preds,
            'int_var_keys': train_set.dataset.int_var_keys,
            'attack_count': config['mttd']['attack_count'],
            'traces_per_attack': config['mttd']['traces_per_attack']
        }
    )
    return module

def construct_loaders(
        train_set: Dataset, val_set: Dataset, test_set: Dataset, config: Dict[str, Any]
) -> Tuple[DataLoader, ...]:
    train_loader = DataLoader(
        train_set, batch_size=config['training']['batch_size'], shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=config['training']['batch_size'], shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=config['training']['batch_size'], shuffle=False, num_workers=4, pin_memory=True
    )
    return train_loader, val_loader, test_loader

def train_model(dest: Path, config: Dict[str, Any]):
    dest.mkdir(exist_ok=True, parents=True)
    train_set, val_set, test_set = construct_datasets(config)
    steps_per_epoch = ceil(len(train_set)/config['training']['batch_size'])
    config['training']['total_steps'] = steps_per_epoch*config['training']['total_epochs']
    config['training']['lr_warmup_steps'] = steps_per_epoch*config['training']['lr_warmup_epochs']
    config['training']['lr_const_steps'] = steps_per_epoch*config['training']['lr_const_epochs']
    train_loader, val_loader, test_loader = construct_loaders(train_set, val_set, test_set, config)
    training_module = construct_training_module(train_set, val_set, test_set, config)
    if config['training']['compile']:
        training_module.model.compile()
    print(training_module.model)
    train_supervised_model(
        dest=dest,
        training_module=training_module,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        total_steps=config['training']['total_steps'],
        grad_clip_val=config['training']['grad_clip_val'],
        accumulate_grad_batches=config['training']['accumulate_grad_batches'],
        early_stop_metric=config['training']['early_stop_metric']
    )

if __name__ == '__main__':
    main(train_model)
