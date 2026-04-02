from typing import Literal, Any, List, Optional, Union, Dict, Annotated, get_args
import math
from pathlib import Path

from pydantic import BaseModel, Field, StrictBool
from scipy.stats import qmc as scipy_qmc
import lightning
import numpy as np
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

def generate_qmc_trials(
        study: optuna.Study,
        search_space: Dict[str, Dict[str, Any]],
        n_trials: int,
        seed: Optional[int] = None,
        qmc_type: str = 'sobol',
) -> None:
    """Pre-generate QMC hyperparameter configurations and enqueue them in the study.

    Each configuration is stored as a WAITING trial via study.enqueue_trial(), so
    parallel workers can call study.optimize(n_trials=1) without any task-ID-to-trial
    mapping.  Because enqueue_trial() is called sequentially here (single process),
    there is no race condition in _find_sample_id.

    Categorical parameters are included as extra QMC dimensions (mapped from [0,1]
    to the discrete choice set) rather than falling back to random sampling.

    Trials that are already WAITING or RUNNING are counted so that re-running this
    script after a partial failure does not double-enqueue configurations.

    Args:
        study: An existing Optuna study.
        search_space: Nested dict matching SupervisedTrainingConfig.search_space,
            e.g. {'model': {'lr': FloatParamConfig(...)}, 'training': {...}}.
        n_trials: Total number of trials to ensure are enqueued.
        seed: Seed for the QMC engine (and for reproducibility).
        qmc_type: 'sobol' or 'halton'.
    """

    # Flatten search_space into an ordered list of (flat_key, ParamConfig) pairs.
    # flat_key is used as the Optuna parameter name, matching what sample_hparams() uses.
    flat_params: List[tuple[str, Any]] = []
    for _field_key, field_space in search_space.items():
        for param_name, param_cfg in field_space.items():
            flat_params.append((param_name, param_cfg))

    d = len(flat_params)
    if d == 0:
        raise ValueError('search_space is empty')

    # Count already-enqueued (WAITING) and in-progress (RUNNING) trials so we
    # don't duplicate them if this script is re-run after a partial failure.
    existing = study.trials
    n_existing = len(existing)
    n_to_add = n_trials - n_existing
    if n_to_add <= 0:
        return

    # Build the QMC engine and advance past already-generated points so the
    # sequence stays consistent on re-runs.
    scramble = seed is not None
    if qmc_type == 'sobol':
        engine = scipy_qmc.Sobol(d=d, scramble=scramble, seed=seed)
    elif qmc_type == 'halton':
        engine = scipy_qmc.Halton(d=d, scramble=scramble, seed=seed)
    else:
        raise ValueError(f'Unknown qmc_type: {qmc_type!r}')

    if n_existing > 0:
        engine.fast_forward(n_existing)

    samples = engine.random(n_to_add)  # shape (n_to_add, d)

    for row in samples:
        params: Dict[str, Any] = {}
        for dim_idx, (param_name, param_cfg) in enumerate(flat_params):
            u = float(row[dim_idx])  # uniform sample in [0, 1)
            if param_cfg.type == 'float':
                if param_cfg.log:
                    val = math.exp(math.log(param_cfg.low) + u * (math.log(param_cfg.high) - math.log(param_cfg.low)))
                else:
                    val = param_cfg.low + u * (param_cfg.high - param_cfg.low)
                if param_cfg.step is not None:
                    val = round((val - param_cfg.low) / param_cfg.step) * param_cfg.step + param_cfg.low
                    val = float(np.clip(val, param_cfg.low, param_cfg.high))
            elif param_cfg.type == 'int':
                if param_cfg.log:
                    val = math.exp(math.log(param_cfg.low) + u * (math.log(param_cfg.high) - math.log(param_cfg.low)))
                    val = int(round(val))
                else:
                    step = param_cfg.step if param_cfg.step is not None else 1
                    n_steps = (param_cfg.high - param_cfg.low) // step
                    val = param_cfg.low + int(math.floor(u * (n_steps + 1))) * step
                    val = int(np.clip(val, param_cfg.low, param_cfg.high))
            elif param_cfg.type == 'categorical':
                choices = param_cfg.choices
                idx = int(math.floor(u * len(choices)))
                idx = min(idx, len(choices) - 1)
                val = choices[idx]
            else:
                raise ValueError(f'Unknown param type: {param_cfg.type!r}')
            params[param_name] = val
        study.enqueue_trial(params)