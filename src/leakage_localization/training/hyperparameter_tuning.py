from typing import Literal, Any, List, Optional, Union, Dict, get_args
from dataclasses import dataclass
from pathlib import Path

import optuna

PARAM = Literal[
    'categorical',
    'float',
    'int'
]

@dataclass
class CategoricalParamConfig:
    type: PARAM = 'categorical'
    choices: List[Any]

    def __post_init__(self):
        assert self.type == 'categorical'
        assert isinstance(self.choices, list)

@dataclass
class FloatParamConfig:
    type: PARAM = 'float'
    low: float
    high: float
    step: Optional[float] = None
    log: bool = False

    def __post_init__(self):
        assert self.type == 'float'
        assert isinstance(self.low, float)
        assert isinstance(self.high, float)
        if self.step is not None:
            assert isinstance(self.step, float) and self.step > 0
        assert isinstance(self.log, bool)

@dataclass
class IntParamConfig:
    type: PARAM = 'int'
    low: int
    high: int
    step: Optional[int] = 1
    log: bool = False

    def __post_init__(self):
        assert self.type == 'int'
        assert isinstance(self.low, int)
        assert isinstance(self.high, int)
        if self.step is not None:
            assert isinstance(self.step, int) and self.step > 0
        assert isinstance(self.log, bool)

ParamConfig = Union[CategoricalParamConfig, FloatParamConfig, IntParamConfig]
SamplerType = Literal[
    'tpe',
    'qmc',
    'random'
]
StudyDirection = Literal[
    'minimize',
    'maximize'
]

def sample_hparams(trial: optuna.Trial, param_configs: Dict[str, ParamConfig]) -> Dict[str, Any]:
    rv = dict()
    for param_key, param_config in param_configs.items():
        if param_config.type == 'categorical':
            param_val = trial.suggest_categorical(name=param_key, choices=param_config.choices)
        elif param_config.type == 'float':
            param_val = trial.suggest_float(name=param_key, low=param_config.low, high=param_config.high, step=param_config.step, log=param_config.log)
        elif param_config.type == 'int':
            param_val = trial.suggest_int(name=param_key, low=param_config.low, high=param_config.high, step=param_config.step, log=param_config.log)
        else:
            assert False
        rv[param_key] = param_val
    return rv

def get_study(
        study_path: Path,
        study_direction: StudyDirection,
        sampler_type: SamplerType = 'random',
        enable_pruning: bool = False,
        seed: Optional[int] = None
) -> optuna.Study:
    assert study_direction in get_args(StudyDirection)

    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(str(study_path))
    )
    if sampler_type == 'tpe':
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=20,
            n_ei_candidates=20,
            multivariate=True,
            group=True,
            constant_liar=True,
            seed=seed
        )
    elif sampler_type == 'qmc':
        sampler = optuna.samplers.QMCSampler(seed=seed)
    elif sampler_type == 'random':
        sampler = optuna.samplers.RandomSampler(seed=seed)
    else:
        assert False
    if enable_pruning:
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=50,
            reduction_factor=2
        )
    else:
        pruner = optuna.pruners.NopPruner()
    study = optuna.create_study(
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        study_name=study_path.stem,
        direction=study_direction,
        load_if_exists=True
    )
    return study