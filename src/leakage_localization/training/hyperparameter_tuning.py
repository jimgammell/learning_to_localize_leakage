from typing import Literal, Any, List, Optional, Union, Dict, Annotated, get_args
from pathlib import Path

from pydantic import BaseModel, Field, StrictBool
import lightning
import optuna

SamplerType = Literal[
    'tpe',
    'qmc',
    'random'
]
StudyDirection = Literal[
    'minimize',
    'maximize'
]

class CategoricalParamConfig(BaseModel):
    type: Literal['categorical'] = 'categorical'
    choices: List[Any]

class FloatParamConfig(BaseModel):
    type: Literal['float'] = 'float'
    low: float
    high: float
    step: Optional[Annotated[float, Field(gt=0)]] = None
    log: StrictBool = False

class IntParamConfig(BaseModel):
    type: Literal['int'] = 'int'
    low: int
    high: int
    step: Optional[Annotated[int, Field(gt=0)]] = 1
    log: StrictBool = False

ParamConfig = Annotated[
    Union[CategoricalParamConfig, FloatParamConfig, IntParamConfig],
    Field(discriminator='type')
]

class PruningCallback(lightning.Callback):
    def __init__(self, trial: optuna.Trial, early_stop_metric: str):
        super().__init__()
        self.trial = trial
        self.early_stop_metric = early_stop_metric

    def on_validation_epoch_end(self, trainer: lightning.Trainer, pl_module: lightning.LightningModule):
        tracked_metric = trainer.callback_metrics[self.early_stop_metric].item()
        self.trial.report(tracked_metric, step=trainer.current_epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()

def sample_hparams(trial: optuna.Trial, param_configs: Dict[str, Any]) -> Dict[str, Any]:
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