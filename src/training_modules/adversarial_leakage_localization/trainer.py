from copy import copy
import json
from math import log, log1p
from collections import defaultdict
from scipy.stats import kendalltau, pearsonr
from torch import nn
from torch.utils.data import Dataset, DataLoader
from lightning import LightningModule, Trainer as LightningTrainer
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from common import *
from trials.utils import *
from datasets.data_module import DataModule
from .module import Module
from .plot_things import *
from utils.dnn_performance_auc import compute_dnn_performance_auc

class Trainer:
    def __init__(self,
        profiling_dataset: Dataset,
        attack_dataset: Dataset,
        default_data_module_kwargs: dict = {},
        default_training_module_kwargs: dict = {},
        reference_leakage_assessment: Optional[np.ndarray] = None
    ):
        self.profiling_dataset = profiling_dataset
        self.attack_dataset = attack_dataset
        self.default_data_module_kwargs = default_data_module_kwargs
        self.default_training_module_kwargs = default_training_module_kwargs
        self.reference_leakage_assessment = reference_leakage_assessment
        
        #self.data_module = DataModule(
        #    self.profiling_dataset,
        #    self.attack_dataset,
        #    **self.default_data_module_kwargs
        #)
    
    def pretrain_classifiers(self,
        logging_dir: Union[str, os.PathLike],
        max_steps: int = 1000,
        override_kwargs: dict = {}
    ):
        if not os.path.exists(os.path.join(logging_dir, 'training_curves.pickle')):
            data_module = DataModule(
                self.profiling_dataset,
                self.attack_dataset,
                **self.default_data_module_kwargs
            )
            if os.path.exists(logging_dir):
                shutil.rmtree(logging_dir)
            os.makedirs(logging_dir)
            kwargs = copy(self.default_training_module_kwargs)
            kwargs.update(override_kwargs)
            training_module = Module(
                self.profiling_dataset.timesteps_per_trace,
                self.profiling_dataset.class_count,
                gamma_bar=0.5,
                train_etat=False,
                **kwargs
            )
            checkpoint = ModelCheckpoint(
                monitor='val_theta_rank',
                mode='min',
                save_top_k=1,
                dirpath=logging_dir,
                filename='best_checkpoint'
            )
            trainer = LightningTrainer(
                max_steps=training_module.to_global_steps(max_steps),
                val_check_interval=1.,
                default_root_dir=logging_dir,
                accelerator='gpu',
                devices=1,
                logger=TensorBoardLogger(logging_dir, name='lightning_output'),
                callbacks=[checkpoint]
            )
            trainer.fit(training_module, datamodule=data_module)
            trainer.save_checkpoint(os.path.join(logging_dir, 'final_checkpoint.ckpt'))
            training_curves = get_training_curves(logging_dir)
            save_training_curves(training_curves, logging_dir)
        plot_training_curves(logging_dir, anim_gammas=False)
    
    def htune_pretrain_classifiers(self,
        logging_dir: Union[str, os.PathLike],
        trial_count: int = 10,
        max_steps: int = 1000,
        override_kwargs: dict = {},
        starting_seed: int = 0
    ):
        lr_vals = [m*10**-4 for m in range(1, 11)]
        results = defaultdict(list)
        for trial_idx in range(trial_count):
            set_seed(starting_seed + trial_idx)
            experiment_dir = os.path.join(logging_dir, f'trial_{trial_idx}')
            os.makedirs(experiment_dir, exist_ok=True)
            hparams = {
                'theta_lr': lr_vals[trial_idx]
            }
            override_kwargs.update(hparams)
            self.pretrain_classifiers(
                logging_dir=experiment_dir,
                max_steps=max_steps, 
                override_kwargs=override_kwargs
            )
            with open(os.path.join(experiment_dir, 'hparams.pickle'), 'wb') as f:
                pickle.dump(hparams, f)
            training_curves = get_training_curves(experiment_dir)
            for key, val in hparams.items():
                results[key].append(val)
            optimal_idx = np.argmin(training_curves['val_theta_rank'][-1])
            results['min_rank'].append(training_curves['val_theta_rank'][-1][optimal_idx])
            results['final_rank'].append(training_curves['val_theta_rank'][-1][-1])
            results['min_loss'].append(training_curves['val_theta_loss'][-1][optimal_idx])
            results['final_loss'].append(training_curves['val_theta_loss'][-1][-1])
        with open(os.path.join(logging_dir, 'results.pickle'), 'wb') as f:
            pickle.dump(results, f)
        best_idx = np.argmin(results['min_rank'])
        best_trial_dir = os.path.join(logging_dir, f'trial_{best_idx}')
        return best_trial_dir
    
    def run(self,
        logging_dir: Union[str, os.PathLike],
        pretrained_classifiers_logging_dir: Optional[Union[str, os.PathLike]] = None,
        max_steps: int = 1000,
        anim_gammas: bool = True,
        override_kwargs: dict = {},
        reference: Optional[np.ndarray] = None,
        ablation='none'
    ):
        if not os.path.exists(os.path.join(logging_dir, 'training_curves.pickle')):
            data_module = DataModule(
                self.profiling_dataset,
                self.attack_dataset,
                **self.default_data_module_kwargs
            )
            if os.path.exists(logging_dir):
                shutil.rmtree(logging_dir)
            os.makedirs(logging_dir)
            kwargs = copy(self.default_training_module_kwargs)
            kwargs.update(override_kwargs)
            training_module = Module(
                self.profiling_dataset.timesteps_per_trace,
                self.profiling_dataset.class_count,
                reference_leakage_assessment=self.reference_leakage_assessment,
                **kwargs
            )
            if pretrained_classifiers_logging_dir is not None:
                assert os.path.exists(pretrained_classifiers_logging_dir)
                pretrained_module = Module.load_from_checkpoint(os.path.join(pretrained_classifiers_logging_dir, 'best_checkpoint.ckpt'))
                training_module.cmi_estimator.classifiers.load_state_dict(pretrained_module.cmi_estimator.classifiers.state_dict())
            trainer = LightningTrainer(
                max_steps=training_module.to_global_steps(max_steps),
                val_check_interval=1.,
                default_root_dir=logging_dir,
                accelerator='gpu',
                devices=1,
                logger=TensorBoardLogger(logging_dir, name='lightning_output'),
                callbacks=[],
                enable_checkpointing=False
            )
            trainer.fit(training_module, datamodule=data_module)
            trainer.save_checkpoint(os.path.join(logging_dir, 'final_checkpoint.ckpt'))
            training_curves = get_training_curves(logging_dir)
            save_training_curves(training_curves, logging_dir)
            if False: # 'supervised_dnn' in override_kwargs:
                training_module = Module.load_from_checkpoint(os.path.join(logging_dir, 'best_checkpoint.ckpt'))
            leakage_assessment = training_module.selection_mechanism.get_gamma().detach().cpu().numpy().reshape(-1)
            np.save(os.path.join(logging_dir, 'leakage_assessment.npy'), leakage_assessment)
            if self.reference_leakage_assessment is not None:
                plot_leakage_assessment(self.reference_leakage_assessment, leakage_assessment, os.path.join(logging_dir, 'leakage_assessment.png'))
        else:
            leakage_assessment = np.load(os.path.join(logging_dir, 'leakage_assessment.npy'))
        training_curves = load_training_curves(logging_dir)
        plot_training_curves(logging_dir, anim_gammas=anim_gammas, reference=reference)
        return leakage_assessment
    
    def htune_leakage_localization(self,
        logging_dir: Union[str, os.PathLike],
        pretrained_classifiers_logging_dir: Optional[Union[str, os.PathLike]] = None,
        trial_count: int = 100,
        max_steps: int = 1000,
        override_kwargs: dict = {},
        ablation = 'none',
        starting_seed: int = 0
    ):
        if ablation in ['two_stage', 'attribution']:
            assert False
        results = defaultdict(list)
        for trial_idx in range(trial_count):
            set_seed(starting_seed + trial_idx)
            experiment_dir = os.path.join(logging_dir, f'trial_{trial_idx}')
            os.makedirs(experiment_dir, exist_ok=True)
            if not os.path.exists(os.path.join(experiment_dir, 'leakage_assessment.npy')):
                hparams = {
                    'theta_lr': float(np.random.choice(sum([[m*10**n for m in range(1, 10)] for n in range(-6, -3)], start=[]))),
                    'gamma_bar': float(np.random.choice(np.arange(0.05, 1.0, 0.05))),
                    'gradient_estimator': np.random.choice(['gumbel'])
                }
                if ablation in ['mask_norm_penalty', 'gamma_norm_penalty']:
                    del hparams['gamma_bar']
                    hparams['norm_penalty_coeff'] = float(np.random.choice(np.logspace(-2, 2, 20)))
                hparams.update({
                    'etat_lr': hparams['theta_lr']*float(np.random.choice(sum([[m*10**n for m in range(1, 10)] for n in range(0, 3)], start=[])))
                })
                override_kwargs.update(hparams)
                leakage_assessment = self.run(
                    experiment_dir, pretrained_classifiers_logging_dir=pretrained_classifiers_logging_dir,
                    max_steps=max_steps, anim_gammas=False, override_kwargs=override_kwargs, ablation=ablation
                )
                with open(os.path.join(experiment_dir, 'hparams.json'), 'w') as f:
                    json.dump(hparams, f, indent='  ')
                np.save(os.path.join(experiment_dir, 'leakage_assessment.npy'), leakage_assessment)
            with open(os.path.join(experiment_dir, 'hparams.json'), 'r') as f:
                hparams = json.load(f)
            for key, val in hparams.items():
                results[key].append(val)
        with open(os.path.join(logging_dir, 'results.pickle'), 'wb') as f:
            pickle.dump(results, f)
    
    def eval_model(self, model_dir: str, pretrained_classifiers_dir: str, output_dir: str, epoch_count: int = 5):
        seed = 0
        set_seed(seed)
        data_module = DataModule(
            self.profiling_dataset,
            self.attack_dataset,
            **self.default_data_module_kwargs
        )
        module = Module.load_from_checkpoint(os.path.join(model_dir, 'final_checkpoint.ckpt'))
        pretrained_classifier_module = Module.load_from_checkpoint(os.path.join(pretrained_classifiers_dir, 'best_checkpoint.ckpt'))
        module.cmi_estimator.load_state_dict(pretrained_classifier_module.cmi_estimator.state_dict())
        val_ranks = []
        for gamma_bar in np.arange(0.05, 1.0, 0.05):
            module.selection_mechanism.gamma_bar = gamma_bar
            nn.init.constant_(module.selection_mechanism.log_C, log(module.hparams.timesteps_per_trace) + log(gamma_bar) - log1p(-gamma_bar))
            trainer = LightningTrainer(
                max_steps=1,
                val_check_interval=1.,
                default_root_dir=output_dir,
                accelerator='gpu',
                devices=1,
                logger=TensorBoardLogger(output_dir, name='lightning_output'),
                callbacks=[],
                enable_checkpointing=False
            )
            trainer.validate(module, datamodule=data_module)
            training_curves = get_training_curves(output_dir)
            val_rank = np.mean(training_curves['val_theta_rank'][-1])
            val_ranks.append(val_rank)
        return val_ranks