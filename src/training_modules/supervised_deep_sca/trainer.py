from copy import copy
import pickle
import json
from collections import defaultdict
from torch.utils.data import Dataset
from lightning import Trainer as LightningTrainer
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from scipy.stats import spearmanr

from common import *
from trials.utils import *
from datasets.data_module import DataModule
from .module import Module
from .plot_things import *
from utils.baseline_assessments.neural_net_attribution import NeuralNetAttribution

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
    
    def run(self,
        logging_dir: Union[str, os.PathLike],
        max_steps: int = 1000,
        override_kwargs: dict = {}
    ):
        data_module = DataModule(
                self.profiling_dataset,
                self.attack_dataset,
                **self.default_data_module_kwargs
            )
        if not os.path.exists(os.path.join(logging_dir, 'training_curves.pickle')):
            if os.path.exists(logging_dir):
                shutil.rmtree(logging_dir)
            os.makedirs(logging_dir)
            kwargs = copy(self.default_training_module_kwargs)
            kwargs.update(override_kwargs)
            print(kwargs)
            training_module = Module(
                timesteps_per_trace=self.profiling_dataset.timesteps_per_trace,
                class_count=self.profiling_dataset.class_count,
                **kwargs
            )
            callbacks = [ModelCheckpoint(
                monitor='val_rank',
                mode='min',
                save_top_k=1,
                dirpath=logging_dir,
                filename='early_stop_checkpoint'
            )]
            trainer = LightningTrainer(
                max_steps=max_steps,
                val_check_interval=1.,
                default_root_dir=logging_dir,
                accelerator='gpu',
                devices=1,
                logger=TensorBoardLogger(logging_dir, name='lightning_output'),
                callbacks=callbacks
            )
            trainer.fit(training_module, datamodule=data_module)
            trainer.save_checkpoint(os.path.join(logging_dir, 'final_checkpoint.ckpt'))
        if not os.path.exists(os.path.join(logging_dir, 'final_leakage_assessments.npz')):
            module = Module.load_from_checkpoint(os.path.join(logging_dir, 'final_checkpoint.ckpt'))
            neural_net_attributor = NeuralNetAttribution(
                data_module.train_dataloader(), module.classifier, device=module.device
            )
            final_leakage_assessments = {
                'gradvis': neural_net_attributor.compute_gradvis().reshape(-1),
                'saliency': neural_net_attributor.compute_saliency().reshape(-1),
                'lrp': neural_net_attributor.compute_lrp().reshape(-1),
                'inputxgrad': neural_net_attributor.compute_inputxgrad().reshape(-1)
            }
            np.savez(os.path.join(logging_dir, 'final_leakage_assessments.npz'), **final_leakage_assessments)
        final_leakage_assessments = np.load(os.path.join(logging_dir, 'final_leakage_assessments.npz'), allow_pickle=True)
        if not os.path.exists(os.path.join(logging_dir, 'early_stop_leakage_assessments.npz')):
            module = Module.load_from_checkpoint(os.path.join(logging_dir, 'early_stop_checkpoint.ckpt'))
            neural_net_attributor = NeuralNetAttribution(
                data_module.train_dataloader(), module.classifier, device=module.device
            )
            early_stop_leakage_assessments = {
                'gradvis': neural_net_attributor.compute_gradvis().reshape(-1),
                'saliency': neural_net_attributor.compute_saliency().reshape(-1),
                'lrp': neural_net_attributor.compute_lrp().reshape(-1),
                'inputxgrad': neural_net_attributor.compute_inputxgrad().reshape(-1)
            }
            np.savez(os.path.join(logging_dir, 'early_stop_leakage_assessments.npz'), **early_stop_leakage_assessments)
        early_stop_leakage_assessments = np.load(os.path.join(logging_dir, 'early_stop_leakage_assessments.npz'), allow_pickle=True)
        training_curves = get_training_curves(logging_dir)
        save_training_curves(training_curves, logging_dir)
        plot_training_curves(logging_dir)
        if self.reference_leakage_assessment is not None:
            rv = {
                **{f'final_{key}': spearmanr(self.reference_leakage_assessment.reshape(-1), assessment.reshape(-1)).statistic for key, assessment in final_leakage_assessments.items()},
                **{f'early_stop_{key}': spearmanr(self.reference_leakage_assessment.reshape(-1), assessment.reshape(-1)).statistic for key, assessment in early_stop_leakage_assessments.items()}
            }
            return rv
    
    def hparam_tune(self,
        logging_dir: Union[str, os.PathLike],
        trial_count: int = 50,
        max_steps: int = 1000,
        override_kwargs: dict = {},
        heavy_input_dropout_ablation: bool = False,
        starting_seed: int = 0
    ):
        lr_vals = sum([[m*10**n for m in range(1, 10)] for n in range(-5, -2)], start=[])
        #weight_decay_vals = [0.0, 1e-6, 1e-4, 1e-2, 1e0]
        lr_scheduler_names = [None, 'CosineDecayLRSched']
        input_dropout_vals = [0.0, 0.1] # np.arange(0.0, 1.0, 0.05) # if heavy_input_dropout_ablation else [0.0, 0.1]
        hidden_dropout_vals = [0.0, 0.2]
        output_dropout_vals = [0.0, 0.3]
        results = defaultdict(list)
        for trial_idx in range(trial_count):
            set_seed(starting_seed + trial_idx)
            experiment_dir = os.path.join(logging_dir, f'trial_{trial_idx}')
            print(f'Running trial {trial_idx} in directory {experiment_dir}...')
            os.makedirs(experiment_dir, exist_ok=True)
            if not os.path.exists(os.path.join(experiment_dir, 'hparams.json')):
                if os.path.exists(os.path.join(experiment_dir, 'hparams.pickle')):
                    with open(os.path.join(experiment_dir, 'hparams.pickle'), 'rb') as f:
                        hparams = pickle.load(f)
                else:
                    hparams = {
                        'lr': np.random.choice(lr_vals),
                        'lr_scheduler_name': np.random.choice(lr_scheduler_names),
                        'input_dropout': np.random.choice(input_dropout_vals),
                        'hidden_dropout': np.random.choice(hidden_dropout_vals),
                        'output_dropout': np.random.choice(output_dropout_vals)
                    }
            else:
                with open(os.path.join(experiment_dir, 'hparams.json'), 'r') as f:
                    hparams = json.load(f)
            override_kwargs.update(hparams)
            rv = self.run(
                logging_dir=experiment_dir,
                max_steps=max_steps,
                override_kwargs=override_kwargs
            )
            training_curves = get_training_curves(experiment_dir)
            for key, val in hparams.items():
                results[key].append(val)
            optimal_idx = np.argmin(training_curves['val_rank'][-1])
            results['min_rank'].append(training_curves['val_rank'][-1][optimal_idx])
            results['final_rank'].append(training_curves['val_rank'][-1][-1])
            results['min_loss'].append(training_curves['val_loss'][-1][optimal_idx])
            results['final_loss'].append(training_curves['val_loss'][-1][-1])
            if rv is not None:
                for key, val in rv.items():
                    results[key].append(val)
            print(f'Done running trial with hparams {hparams}.')
            for key, val in results.items():
                print(f'\t{key}={val[-1]}')
            with open(os.path.join(experiment_dir, 'hparams.json'), 'w') as f:
                json.dump(hparams, f, indent='  ')
        with open(os.path.join(logging_dir, 'results.pickle'), 'wb') as f:
            pickle.dump(results, f)