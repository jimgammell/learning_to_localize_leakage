from pathlib import Path
from typing import Dict, Any, List, Optional
from functools import partial
from copy import deepcopy
import os

import optuna

from experiments.initialization import SEED

def sample_hparams(
        trial: optuna.Trial,
        base_config: Dict[str, Any]
) -> Dict[str, Any]:
    config = deepcopy(base_config)
    for section_key, section in config['search_space'].items():
        for hparam_key, hparam_config in section.items():
            type = hparam_config['type']
            if type == 'categorical':
                choices: List[Any] = hparam_config['choices']
                sample_fn = partial(trial.suggest_categorical, choices=choices)
            elif type == 'float':
                low: float = hparam_config['low']
                high: float = hparam_config['high']
                step: Optional[float] = hparam_config.get('step', None)
                log: bool = hparam_config.get('log', False)
                sample_fn = partial(trial.suggest_float, low=low, high=high, step=step, log=log)
            elif type == 'int':
                low: int = hparam_config['low']
                high: int = hparam_config['high']
                step: int = hparam_config.get('step', 1)
                log: bool = hparam_config.get('log', False)
                sample_fn = partial(trial.suggest_int, low=low, high=high, step=step, log=log)
            else:
                assert False
            config[section_key][hparam_key] = sample_fn(name=f'{section_key}.{hparam_key}')
    return config

def get_study(
        study_path: Path,
        config: Dict[str, Any]
) -> optuna.Study:
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(str(study_path))
    )
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=48,
        n_ei_candidates=24,
        multivariate=True,
        group=True,
        constant_liar=True,
        seed=SEED + int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    )
    # I don't think pruning makes sense on a lot of these datasets since it takes so long to break random performance
    #pruner = optuna.pruners.HyperbandPruner(
    #    min_resource=50,
    #    reduction_factor=2
    #)
    study = optuna.create_study(
        storage=storage,
        sampler=sampler,
        #pruner=pruner,
        study_name=study_path.stem,
        direction={'min': 'minimize', 'max': 'maximize'}[config['training']['early_stop_mode']],
        load_if_exists=True
    )
    return study