from typing import Dict, Any, Tuple, List
from copy import copy
from math import ceil

import numpy as np
from torch.utils.data import Dataset, Subset
import lightning
from leakage_localization.datasets import Base_TorchDataset
from leakage_localization.training import SupervisedModule
from leakage_localization.models import Model

from init_things import *
from utils.load_data import load_torch_dataset, construct_loaders

def construct_datasets(config: Dict[str, Any]) -> Tuple[Dataset, ...]:
    profiling_set = load_torch_dataset(
        config['data']['id'],
        'profile',
        target_byte=config['data']['target_byte'],
        target_variable=config['data']['target_variable']
    )
    attack_set = load_torch_dataset(
        config['data']['id'],
        'attack',
        target_byte=config['data']['target_byte'],
        target_variable=config['data']['target_variable']
    )
    indices = np.random.default_rng(seed=str_to_seed('data_partition')).permutation(len(profiling_set))
    val_len = int(len(indices)*config['data']['val_prop'])
    train_set = Subset(copy(profiling_set), indices=indices[:-val_len])
    val_set = Subset(copy(profiling_set), indices=indices[-val_len:])
    test_set = copy(attack_set)
    return profiling_set, attack_set, train_set, val_set, test_set

def construct_module(profiling_set: Base_TorchDataset, config: Dict[str, Any]) -> SupervisedModule:
    module = SupervisedModule(
        model_constructor=Model,
        model_kwargs=dict(
            input_length=profiling_set.timestep_count,
            output_count=(
                66 if config['model']['grey_box_head'] == 'ascadv1' else
                18 if config['model']['grey_box_head'] == 'ascadv2' else
                profiling_set.config.num_labels
            ),
            grey_box_head=config['model']['grey_box_head'],
            trunk=config['model']['trunk'],
            position_embedding=config['model']['position_embedding'],
            pooling=config['model']['pooling'],
            head=config['model']['head'],
            fnn_style=config['model']['fnn_style'],
            patch_size=config['model']['patch_size'],
            use_fourier_embed=config['model']['use_fourier_embed'],
            fourier_embed_num_bands=config['model']['fourier_embed_num_bands'],
            fourier_embed_sigma=config['model']['fourier_embed_sigma'],
            embedding_dim=config['model']['embedding_dim'],
            trunk_blocks=config['model']['trunk_blocks'],
            head_blocks=config['model']['head_count'],
            register_tokens=config['model']['register_tokens'],
            input_dropout_rate=config['model']['input_dropout_rate'],
            input_droppatch_rate=config['model']['input_droppatch_rate'],
            hidden_dropout_rate=config['model']['hidden_dropout_rate'],
            use_bias=config['model']['use_bias'],
            perceiver_latent_dim=config['model']['perceiver_latent_dim'],
            perceiver_self_attn_per_cross_attn_blocks=config['model']['perceiver_self_attn_per_cross_attn_blocks'],
            perceiver_cross_attn_head_count=config['model']['perceiver_cross_attn_head_count']
        ),
        leakage_model=config['model']['leakage_model'],
        num_labels=profiling_set.config.num_labels,
        total_steps=config['training']['total_steps'],
        lr_warmup_steps=config['training']['lr_warmup_steps'],
        lr_const_steps=config['training']['lr_const_steps'],
        base_lr=config['training']['base_lr'],
        lr_decay_multiplier=config['training']['lr_decay_multiplier'],
        weight_decay=config['training']['weight_decay'],
        label_smoothing=config['training']['label_smoothing'],
        mtd_kwargs={
            'target_preds_to_key_preds': profiling_set.target_preds_to_key_preds,
            'int_var_keys': profiling_set.int_var_keys,
            'attack_count': config['mtd']['attack_count'],
            'traces_per_attack': config['mtd']['traces_per_attack']
        },
        trace_statistics=profiling_set.get_trace_statistics(),
        additive_gaussian_noise=config['training']['additive_gaussian_noise'],
        mixup_alpha=config['training']['mixup_alpha'],
        preprocessing=config['data']['preprocessing'],
        random_roll_scale=float(config['data']['random_roll_scale']),
        random_lpf_scale=float(config['data']['random_lpf_scale']),
        compute_val_mtd=config['training']['early_stop_metric'] == 'val/mtd'
    )
    return module

def run_train_model(
        dest: Path,
        config: Dict[str, Any],
        aux_callbacks: Optional[List[lightning.Callback]] = None
):
    dest.mkdir(exist_ok=True, parents=True)
    if 'seed' in config['training']:
        set_seed(config['training']['seed'])
    profiling_set, attack_set, train_set, val_set, test_set = construct_datasets(config)
    steps_per_epoch = ceil(len(train_set)/config['training']['batch_size'])
    if 'total_steps' not in config['training']:
        config['training']['total_steps'] = steps_per_epoch*config['training']['total_epochs']