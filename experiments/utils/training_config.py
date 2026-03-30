from dataclasses import dataclass
from typing import List, Union, Optional, Dict, Any, get_args
from collections import defaultdict
import subprocess
from pathlib import Path

import yaml

from leakage_localization.datasets import DATASET
from leakage_localization.training.supervised_lightning_module import PREPROCESSING, LEAKAGE_MODEL
from leakage_localization.training.hyperparameter_tuning import CategoricalParamConfig, FloatParamConfig, IntParamConfig, ParamConfig
from leakage_localization.models.model import GREY_BOX_HEAD, TRUNK, POSITION_EMBEDDING, POOLING, HEAD, FNN_STYLE

@dataclass
class DataConfig:
    id: DATASET
    target_byte: Union[int, List[int]]
    target_variable: str
    preprocessing: PREPROCESSING
    random_roll_scale: float
    random_lpf_scale: float
    val_prop: float

    def __post_init__(self):
        assert self.id in get_args(DATASET)
        if isinstance(self.target_byte, int):
            self.target_byte = [self.target_byte]
        assert isinstance(self.target_byte, list) and all(isinstance(x, int) for x in self.target_byte)
        # would be nice to figure out a way to validate the target variable
        assert self.preprocessing in get_args(PREPROCESSING)
        assert isinstance(self.random_roll_scale, float) and self.random_roll_scale >= 0
        assert isinstance(self.random_lpf_scale, float) and self.random_lpf_scale >= 0
        assert isinstance(self.val_prop, float) and 0 < self.val_prop < 1

@dataclass
class TrainingConfig:
    total_steps: int
    lr_warmup_frac: float
    lr_const_frac: float
    batch_size: int
    base_lr: float
    lr_decay_multiplier: float
    weight_decay: float
    label_smoothing: float
    mixup_alpha: float
    additive_gaussian_noise: float
    grad_clip_val: Optional[float]
    accumulate_grad_batches: int
    early_stop_metric: str
    early_stop_mode: str
    seed: int
    compile: bool
    num_workers: int

    def __post_init__(self):
        assert isinstance(self.total_steps, int) and self.total_steps > 0
        assert isinstance(self.lr_warmup_frac, float) and 0 <= self.lr_warmup_frac <= 1
        assert isinstance(self.lr_const_frac, float) and 0 <= self.lr_const_frac <= 1
        assert isinstance(self.batch_size, int) and self.batch_size > 0
        assert isinstance(self.base_lr, float) and self.base_lr > 0
        assert isinstance(self.lr_decay_multiplier, float) and 0 <= self.lr_decay_multiplier <= 1
        assert isinstance(self.weight_decay, float) and self.weight_decay >= 0
        assert isinstance(self.label_smoothing, float) and 0 <= self.label_smoothing < 1
        assert isinstance(self.mixup_alpha, float) and self.mixup_alpha >= 0
        assert isinstance(self.additive_gaussian_noise, float) and self.additive_gaussian_noise >= 0
        if self.grad_clip_val is not None:
            assert isinstance(self.grad_clip_val, float) and self.grad_clip_val > 0
        assert isinstance(self.accumulate_grad_batches, int) and self.accumulate_grad_batches > 0
        assert isinstance(self.early_stop_metric, str)
        assert self.early_stop_mode in {'min', 'max'}
        assert isinstance(self.seed, int)
        assert isinstance(self.compile, bool)
        assert isinstance(self.num_workers, int) and self.num_workers >= 0

@dataclass
class MTDConfig:
    attack_count: int
    traces_per_attack: int

    def __post_init__(self):
        assert isinstance(self.attack_count, int) and self.attack_count > 0
        assert isinstance(self.traces_per_attack, int) and self.traces_per_attack > 0

@dataclass
class ModelConfig:
    grey_box_head: Optional[GREY_BOX_HEAD]
    trunk: TRUNK
    position_embedding: POSITION_EMBEDDING
    pooling: POOLING
    head: HEAD
    fnn_style: FNN_STYLE
    patch_size: Optional[int]
    use_fourier_embed: bool
    fourier_embed_num_bands: Optional[int]
    fourier_embed_sigma: Optional[float]
    embedding_dim: int
    expansion_factor: int
    trunk_blocks: int
    head_count: Optional[int]
    register_tokens: int
    input_dropout_rate: float
    input_droppatch_rate: float
    hidden_dropout_rate: float
    use_bias: bool
    perceiver_latent_dim: Optional[int]
    perceiver_self_attn_per_cross_attn_blocks: Optional[int]
    perceiver_cross_attn_head_count: Optional[int]
    leakage_model: LEAKAGE_MODEL

    def __post_init__(self):
        if self.grey_box_head is not None:
            assert self.grey_box_head in get_args(GREY_BOX_HEAD)
        assert self.trunk in get_args(TRUNK)
        assert self.position_embedding in get_args(POSITION_EMBEDDING)
        assert self.pooling in get_args(POOLING)
        assert self.head in get_args(HEAD)
        assert self.fnn_style in get_args(FNN_STYLE)
        if self.patch_size is not None:
            assert isinstance(self.patch_size, int) and self.patch_size > 0
        assert isinstance(self.use_fourier_embed, bool)
        if self.use_fourier_embed:
            assert isinstance(self.fourier_embed_num_bands, int) and self.fourier_embed_num_bands > 0
            assert isinstance(self.fourier_embed_sigma, float) and self.fourier_embed_sigma > 0
        else:
            assert self.fourier_embed_num_bands is None
            assert self.fourier_embed_sigma is None
        assert isinstance(self.embedding_dim, int) and self.embedding_dim > 0
        assert isinstance(self.expansion_factor, int) and self.expansion_factor > 0
        assert isinstance(self.trunk_blocks, int) and self.trunk_blocks > 0
        if self.head_count is not None:
            assert isinstance(self.head_count, int) and self.head_count > 0 and self.embedding_dim % self.head_count == 0
        assert isinstance(self.register_tokens, int) and self.register_tokens >= 0
        assert isinstance(self.input_dropout_rate, float) and self.input_dropout_rate >= 0
        assert isinstance(self.input_droppatch_rate, float) and self.input_droppatch_rate >= 0
        assert isinstance(self.hidden_dropout_rate, float) and self.hidden_dropout_rate >= 0
        assert isinstance(self.use_bias, bool)
        if self.trunk == 'perceiver':
            assert isinstance(self.perceiver_latent_dim, int) and self.perceiver_latent_dim > 0
            assert isinstance(self.perceiver_self_attn_per_cross_attn_blocks, int) and self.perceiver_self_attn_per_cross_attn_blocks > 0
            if self.perceiver_cross_attn_head_count is not None:
                assert isinstance(self.perceiver_cross_attn_head_count, int) and self.perceiver_cross_attn_head_count > 0
                assert self.perceiver_latent_dim % self.perceiver_cross_attn_head_count == 0
        else:
            assert self.perceiver_latent_dim is None
            assert self.perceiver_self_attn_per_cross_attn_blocks is None
            assert self.perceiver_cross_attn_head_count is None
        assert self.leakage_model in get_args(LEAKAGE_MODEL)

def construct_search_space(search_space_kw: Dict[str, Dict[str, Any]]):
    search_space = defaultdict(dict)
    for field_key, field_search_space in search_space_kw.items():
        for param_key, param_config in field_search_space.items():
            assert 'type' in param_config
            param_type = param_config['type']
            if param_type == 'categorical':
                search_space[field_key][param_key] = CategoricalParamConfig(**param_config)
            elif param_type == 'float':
                search_space[field_key][param_key] = FloatParamConfig(**param_config)
            elif param_type == 'int':
                search_space[field_key][param_key] = IntParamConfig(**param_config)
            else:
                assert False
    return search_space

@dataclass
class SupervisedTrainingConfig:
    data: DataConfig
    training: TrainingConfig
    mtd: MTDConfig
    model: ModelConfig
    search_space: Dict[str, Dict[str, ParamConfig]]
    commit_hash: str = None

    def __post_init__(self):
        self.data = DataConfig(**self.data)
        self.training = TrainingConfig(**self.training)
        self.mtd = MTDConfig(**self.mtd)
        self.model = ModelConfig(**self.model)
        self.search_space = construct_search_space(self.search_space)
        assert self.commit_hash == subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()