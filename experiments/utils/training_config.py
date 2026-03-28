from dataclasses import dataclass
from typing import List, Union, Optional, Dict, Any, get_args
from collections import defaultdict

from leakage_localization.datasets import DATASET
from leakage_localization.training.supervised_lightning_module import PREPROCESSING
from leakage_localization.models.model import ModelConfig
from leakage_localization.training.hyperparameter_tuning import CategoricalParamConfig, FloatParamConfig, IntParamConfig

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
        assert id in get_args(DATASET)
        if isinstance(self.target_byte, int):
            self.target_byte = [self.target_byte]
        assert isinstance(self.target_byte, list) and all(isinstance(x, int) for x in self.target_byte)
        # figure out a way to validate the target variable
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
class SupervisedTrainingConfig:
    data: Dict[str, Any]
    training: Dict[str, Any]
    mtd: Dict[str, Any]
    model: Dict[str, Any]
    search_space: Dict[str, Dict[str, Any]]

    def __post_init__(self):
        self.data = DataConfig(**self.data)
        self.training = TrainingConfig(**self.training)
        self.mtd = MTDConfig(**self.mtd)
        self.model = ModelConfig(**self.model)
        search_space = defaultdict(dict)
        for field_key, field_search_space in self.search_space.items():
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
        self.search_space = search_space