from typing import Literal, Dict, Any, Optional
import os
import shutil
from copy import copy
import pickle
import json
from tqdm import tqdm

import numpy as np
from scipy.stats import spearmanr
from torch.utils.data import Dataset, DataLoader

from common import *
from utils.baseline_assessments import FirstOrderStatistics, NeuralNetAttribution, OccPOI
from utils import dnn_performance_auc
from .plot_things import *
from datasets.data_module import DataModule
from training_modules.supervised_deep_sca import SupervisedTrainer, SupervisedModule
from training_modules.adversarial_leakage_localization import ALLTrainer, ALLModule
from . import supervised_experiment_methods
from . import evaluation_methods
from . import all_experiment_methods

OPTIMAL_WINDOW_SIZES = {
    'ascadv1-fixed': 3,
    'ascadv1-variable': 7,
    'dpav4': 41,
    'aes-hd': 31,
    'otiait': 3,
    'otp': 5
}

class Trial:
    def __init__(self,
        dataset_name: Literal['ascadv1-fixed', 'ascadv1-variable', 'dpav4', 'aes-hd', 'otiait', 'otp'],
        trial_config: Dict[str, Any],
        seed_count: int = 1,
        logging_dir: Optional[str] = None,
        run_particular_seeds: List[int] = []
    ):
        self.dataset_name = dataset_name
        self.trial_config = trial_config
        self.seed_count = seed_count
        self.logging_dir = logging_dir or dataset_name.replace('-', '_')
        self.run_particular_seeds = run_particular_seeds
        if len(self.run_particular_seeds) == 0:
            self.run_particular_seeds = list(range(self.seed_count))
        self.run_particular_seeds = np.array(self.run_particular_seeds)
        if 'data_dir' in self.trial_config:
            self.data_dir = self.trial_config['data_dir']
        else:
            self.data_dir = self.trial_config['dataset']
        if not self.data_dir[0] == os.sep:
            self.data_dir = os.path.join(RESOURCE_DIR, self.data_dir)

        print(f'Initializing trial with dataset `{self.dataset_name}`.')
        print(f'\tSeed count: {self.seed_count}')
        print(f'\tLogging directory: `{self.logging_dir}`')
        print('Trial configuration settings:')
        for key, val in self.trial_config.items():
            print(f'\t{key}={val}')

        self.random_assessment_dir = os.path.join(self.logging_dir, 'random_assessment')
        self.first_order_parametric_stats_dir = os.path.join(self.logging_dir, 'first_order_parametric_statistical_assessment')
        self.supervised_hparam_sweep_dir = os.path.join(self.logging_dir, 'supervised_hparam_sweep')
        self.supervised_attribution_dir = os.path.join(self.logging_dir, 'supervised_models_for_attribution')
        self.supervised_selection_dir = os.path.join(self.logging_dir, 'supervised_models_for_model_selection')
        self.all_hparam_sweep_dir = os.path.join(self.logging_dir, 'all_hparam_sweep')
        self.all_classifiers_pretrain_dir = os.path.join(self.logging_dir, 'all_classifiers_pretrain')
        self.all_dir = os.path.join(self.logging_dir, 'all_runs')
        self.all_sensitivity_analysis_dir = os.path.join(self.logging_dir, 'all_sensitivity_analysis')
        self.plots_for_paper_dir = os.path.join(self.logging_dir, 'plots_for_paper')
        self.attr_over_time_dir = os.path.join(self.logging_dir, 'attr_over_time')

        if self.dataset_name in ['ascadv1-fixed', 'ascadv1-variable']:
            self.oracle_targets = [ # based on Egger (2021) findings
                'r_in', 'r', 'r_out', 'plaintext__key__r_in', 'subbytes__r', 'subbytes__r_out', 's_prev__subbytes__r_out', 'security_load'
            ]
        elif self.dataset_name in ['dpav4', 'aes-hd', 'otiait', 'otp']: # since these are unprotected, I'm only considering first-order leakage
            self.oracle_targets = ['label']
        else:
            assert False
    
    def construct_datasets(self):
        print('Constructing datasets.')
        if self.dataset_name == 'dpav4':
            from datasets.dpav4 import DPAv4
            self.profiling_dataset = DPAv4(root=self.data_dir, train=True)
            self.attack_dataset = DPAv4(root=self.data_dir, train=False)
        elif self.dataset_name == 'ascadv1-fixed':
            from datasets.ascadv1 import ASCADv1
            self.profiling_dataset = ASCADv1(root=self.data_dir, variable_keys=False, train=True)
            self.attack_dataset = ASCADv1(root=self.data_dir, variable_keys=False, train=False)
        elif self.dataset_name == 'ascadv1-variable':
            from datasets.ascadv1 import ASCADv1
            self.profiling_dataset = ASCADv1(root=self.data_dir, variable_keys=True, train=True)
            self.attack_dataset = ASCADv1(root=self.data_dir, variable_keys=True, train=False)
        elif self.dataset_name == 'aes-hd':
            from datasets.aes_hd import AES_HD
            self.profiling_dataset = AES_HD(root=self.data_dir, train=True)
            self.attack_dataset = AES_HD(root=self.data_dir, train=False)
        elif self.dataset_name == 'otiait':
            from datasets.ed25519_wolfssl import ED25519
            self.profiling_dataset = ED25519(root=self.data_dir, train=True)
            self.attack_dataset = ED25519(root=self.data_dir, train=False)
        elif self.dataset_name == 'otp':
            from datasets.one_truth_prevails import OneTruthPrevails
            self.profiling_dataset = OneTruthPrevails(root=self.data_dir, train=True)
            self.attack_dataset = OneTruthPrevails(root=self.data_dir, train=False)
        else:
            assert False
    
    # Random baseline
    def compute_random_assessment(self):
        os.makedirs(self.random_assessment_dir, exist_ok=True)
        if not os.path.exists(os.path.join(self.random_assessment_dir, 'assessment.npy')):
            print('Computing random assessment.')
            assessments = []
            for seed in range(self.seed_count):
                set_seed(seed)
                assessment = np.random.randn(self.profiling_dataset.timesteps_per_trace)
                assessments.append(assessment)
            np.save(os.path.join(self.random_assessment_dir, 'assessment.npy'), np.stack(assessments))
        else:
            print('Random assessment already exists.')
            assessment = np.load(os.path.join(self.random_assessment_dir, 'assessment.npy'))
            seed_count, timestep_count = assessment.shape
            assert timestep_count == self.profiling_dataset.timesteps_per_trace
            if seed_count < self.seed_count:
                print(f'\tNot enough seeds. Running {self.seed_count-seed_count} more trials.')
                for seed in range(seed_count, self.seed_count):
                    set_seed(seed)
                    assessment = np.concatenate([np.random.randn(1, timestep_count), assessment], axis=0)
                np.save(os.path.join(self.random_assessment_dir, 'assessment.npy'), assessment)
    
    def load_random_assessment(self) -> Optional[np.ndarray]:
        if not os.path.exists(os.path.join(self.random_assessment_dir, 'assessment.npy')):
            return None
        else:
            return np.load(os.path.join(self.random_assessment_dir, 'assessment.npy'))
    
    def plot_random_assessment(self):
        random_assessment = self.load_random_assessment()
        oracle_assessment = self.load_oracle_assessment()
        plot_leakage_assessment(oracle_assessment, random_assessment, os.path.join(self.random_assessment_dir, 'visualization.png'), self.dataset_name)
    
    # Parametric statistical methods. Here we are computing both the baseline methods and the oracle methods for calculating agreement with oSNR.
    def compute_first_order_parametric_stats(self):
        os.makedirs(self.first_order_parametric_stats_dir, exist_ok=True)
        profiling_dataset = self.profiling_dataset
        if self.dataset_name == 'dpav4': # 'canonical' attack dataset is too small to get decent measurements -- I repartitioned it into a 3k/2k profile/attack split
            from datasets.dpav4 import DPAv4
            attack_dataset = DPAv4(root=self.data_dir, train=False, ground_truth=True)
        else:
            attack_dataset = self.attack_dataset
        for target in list(set(['label'] + self.oracle_targets)):
            stats_computer = None
            def _compute_and_save(method: Literal['snr', 'sosd', 'cpa'], split: Literal['profiling', 'attack']):
                path = os.path.join(self.first_order_parametric_stats_dir, f'{split}_{method}_{target}.npy')
                if not os.path.exists(path):
                    print(f'Computing {split} {method.upper()} for target `{target}`')
                    nonlocal stats_computer
                    dataset = profiling_dataset if split == 'profiling' else attack_dataset if split == 'attack' else None
                    stats_computer = stats_computer or FirstOrderStatistics(dataset, target)
                    np.save(path, getattr(stats_computer, f'{method}_vals')[target].reshape(-1))
                else:
                    print(f'Found existing {split} {method.upper()} for target `{target}`')
            for method in ['snr', 'sosd', 'cpa']:
                _compute_and_save(method, 'profiling')
            stats_computer = None
            for method in ['snr', 'sosd', 'cpa']:
                _compute_and_save(method, 'attack')
    
    def load_first_order_parametric_assessments(self) -> Dict[str, np.ndarray]:
        rv = {}
        for method in ['snr', 'sosd', 'cpa']:
            path = os.path.join(self.first_order_parametric_stats_dir, f'profiling_{method}_label.npy')
            if os.path.exists(path):
                assessment = np.load(path)
            else:
                assessment = None
            rv[method] = assessment
        return rv
    
    def load_oracle_assessment(self, reduce=True) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        rv = {}
        for target in self.oracle_targets:
            path = os.path.join(self.first_order_parametric_stats_dir, f'attack_snr_{target}.npy')
            assessment = np.load(path)
            rv[target] = assessment
        if reduce:
            rv = np.mean(np.stack(list(rv.values())), axis=0)
        return rv

    def plot_assessment_vs_oracle_leakage(self, assessment, dest):
        if not self.dataset_name in ['ascadv1-fixed', 'ascadv1-variable']:
            return
        oracle_assessments = self.load_oracle_assessment(reduce=False)
        sorted_indices = assessment.argsort()
        assessment = assessment[sorted_indices]
        oracle_assessments = {key: val[sorted_indices] for key, val in oracle_assessments.items()}
        fig, axes = plt.subplots(2, 4, figsize=(4*PLOT_WIDTH, 2*PLOT_WIDTH))
        for (target_name, target_snr), ax in zip(oracle_assessments.items(), axes.flatten()):
            ax.plot(assessment, target_snr, linestyle='none', marker='.', markersize=2, label=target_name, **PLOT_KWARGS)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_title(target_name)
        #ax.legend()
        #ax.set_xlabel('Assessment under test')
        #ax.set_ylabel('Oracle SNR')
        fig.tight_layout()
        fig.savefig(dest, **SAVEFIG_KWARGS)
        plt.close(fig)
    
    def compute_oracle_snr_corr(self, leakage_assessment):
        oracle_assessment = self.load_oracle_assessment()
        corr = spearmanr(leakage_assessment.reshape(-1), oracle_assessment.reshape(-1)).statistic
        if np.isnan(corr):
            corr = 0.0
        return corr
    
    def compute_dnno_auc(self, leakage_assessment, supervised_model=None, seed: int = 0):
        if supervised_model is None:
            supervised_model = self.load_supervised_model(os.path.join(self.supervised_selection_dir, f'seed={seed}'), 'early_stop')
        set_seed(seed)
        data_module = DataModule(self.profiling_dataset, self.attack_dataset)
        val_dataloader = data_module.val_dataloader()
        auc_rv = dnn_performance_auc.compute_dnn_performance_auc(
            val_dataloader, supervised_model, leakage_assessment,
            device='cuda' if torch.cuda.is_available() else 'cpu', cluster_count=None
        )
        fwd_dnno = auc_rv['forward_dnn_auc']
        rev_dnno = auc_rv['reverse_dnn_auc']
        return fwd_dnno, rev_dnno
    
    def get_dataloader(self, split: Literal['train', 'val', 'profile', 'attack'] = 'train', seed=0):
        set_seed(seed)
        data_module = DataModule(self.profiling_dataset, self.attack_dataset)
        if split == 'train':
            return data_module.train_dataloader()
        elif split == 'val':
            return data_module.val_dataloader()
        elif split == 'profile':
            return data_module.profiling_dataloader()
        elif split == 'attack':
            return data_module.test_dataloader()
        else:
            assert False
    
    def eval_all_assessment(self, assessment_dir, seed: int = 0):
        output_dir = os.path.join(assessment_dir, 'eval_output')
        kwargs = copy(self.trial_config['default_kwargs'])
        kwargs.update(self.trial_config['classifiers_pretrain_kwargs'])
        kwargs.update(self.trial_config['all_kwargs'])
        trainer = ALLTrainer(self.profiling_dataset, self.attack_dataset, default_training_module_kwargs=kwargs)
        res = trainer.eval_model(assessment_dir, self.find_best_all_classifiers_pretrain_dir(), output_dir)
        return res
    
    def plot_first_order_parametric_assessments(self):
        first_order_parametric_assessments = self.load_first_order_parametric_assessments()
        oracle_assessment = self.load_oracle_assessment()
        for method, assessment in first_order_parametric_assessments.items():
            plot_leakage_assessment(
                oracle_assessment, assessment, 
                os.path.join(self.first_order_parametric_stats_dir, f'{method}_visualization.png'), self.dataset_name
            )
        
    def plot_oracle_assessments(self):
        oracle_assessments = self.load_oracle_assessment(reduce=False)
        plot_oracle_assessment(oracle_assessments, os.path.join(self.first_order_parametric_stats_dir, 'oracle_snr_visualization.png'), self.dataset_name)
    
    def run_supervised_trials(self):
        base_seed = 0
        base_supervised_kwargs = copy(self.trial_config['default_kwargs'])
        base_supervised_kwargs.update(self.trial_config['supervised_training_kwargs'])
        supervised_experiment_methods.run_supervised_hparam_sweep(
            self.supervised_hparam_sweep_dir, self.profiling_dataset, self.attack_dataset, training_kwargs=base_supervised_kwargs,
            trial_count=self.trial_config['supervised_htune_trial_count'], max_steps=self.trial_config['supervised_train_steps'], starting_seed=base_seed
        )
        base_seed += self.trial_config['supervised_htune_trial_count']
        best_supervised_hparams = supervised_experiment_methods.get_best_supervised_model_hparams(
            self.supervised_hparam_sweep_dir, self.profiling_dataset, self.attack_dataset, self.dataset_name, self.load_oracle_assessment()
        )
        for seed in base_seed + self.run_particular_seeds:
            best_classification_kwargs = copy(base_supervised_kwargs)
            best_classification_kwargs.update(best_supervised_hparams['classification'])
            supervised_experiment_methods.train_supervised_model(
                os.path.join(self.supervised_selection_dir, f'seed={seed}'), self.profiling_dataset, self.attack_dataset, training_kwargs=best_classification_kwargs,
                max_steps=self.trial_config['supervised_train_steps'], seed=seed
            )
            base_seed += 1
        r"""print('Running extra trials for last-minute issues')
        for seed in base_seed + self.run_particular_seeds:
            best_classification_kwargs = copy(base_supervised_kwargs)
            best_classification_kwargs.update(best_supervised_hparams['classification'])
            subdir = os.path.join(self.attr_over_time_dir, f'seed={seed}')
            supervised_experiment_methods.train_supervised_model(
                subdir, self.profiling_dataset, self.attack_dataset, training_kwargs=best_classification_kwargs, dataset_name=self.dataset_name,
                max_steps=self.trial_config['supervised_train_steps'], seed=seed, reference_leakage_assessment=self.load_oracle_assessment()
            )
            if self.dataset_name in ['dpav4', 'aes-hd']:
                supervised_experiment_methods.attribute_neural_net(
                    subdir, self.profiling_dataset, self.attack_dataset, self.dataset_name,
                    compute_gradvis=False, compute_saliency=False, compute_inputxgrad=False,
                    compute_lrp=False, compute_occlusion=np.arange(1, 51, 2), compute_second_order_occlusion=[],
                    compute_occpoi=False, compute_extended_occpoi=False
                )
            base_seed += 1"""
        for name, hparams in best_supervised_hparams.items():
            if name != 'classification': # FIXME
                continue
            model_dir = os.path.join(self.supervised_attribution_dir, name)
            print(f'Running experiments for {model_dir} with hparams {hparams}')
            for seed in range(base_seed, base_seed + self.seed_count):
                subdir = os.path.join(model_dir, f'seed={seed}')
                kwargs = copy(base_supervised_kwargs)
                kwargs.update(hparams)
                supervised_experiment_methods.train_supervised_model(
                    subdir, self.profiling_dataset, self.attack_dataset, training_kwargs=kwargs,
                    max_steps=self.trial_config['supervised_train_steps'], seed=seed, reference_leakage_assessment=self.load_oracle_assessment()
                )
                supervised_experiment_methods.eval_on_attack_dataset(subdir, self.profiling_dataset, self.attack_dataset, self.dataset_name)
                supervised_experiment_methods.attribute_neural_net(
                    subdir, self.profiling_dataset, self.attack_dataset, self.dataset_name,
                    compute_gradvis=True, compute_saliency=True, compute_inputxgrad=True,
                    compute_lrp=True, compute_occlusion=np.arange(1, 51, 2), compute_second_order_occlusion=[1, OPTIMAL_WINDOW_SIZES[self.dataset_name]],
                    compute_occpoi=True, compute_extended_occpoi=False
                )
                if name == 'classification': # compute DNN occlusion tests
                    eval_model = supervised_experiment_methods.load_trained_supervised_model(os.path.join(self.supervised_selection_dir, f'seed={seed-5}'))
                    dataloader = supervised_experiment_methods.get_dataloader(self.profiling_dataset, self.attack_dataset, split='attack')
                    for method_name in ['gradvis', 'inputxgrad', 'lrp', 'occpoi', 'saliency', '1-second-order-occlusion', *[f'{m}-occlusion' for m in np.arange(1, 51, 2)]]:
                        path = os.path.join(self.supervised_attribution_dir, 'classification', f'seed={seed}', f'{method_name}_dnno.npz')
                        if not os.path.exists(path):
                            assessment = np.load(os.path.join(self.supervised_attribution_dir, 'classification', f'seed={seed}', f'{method_name}.npz'), allow_pickle=True)['attribution']
                            metric = 'traces_to_disclosure' if self.dataset_name in ['ascadv1-fixed', 'ascadv1-variable', 'dpav4', 'aes-hd'] else 'mean_rank'
                            fwd_dnno = evaluation_methods.get_forward_dnno_criterion(assessment, eval_model, dataloader)
                            rev_dnno = evaluation_methods.get_reverse_dnno_criterion(assessment, eval_model, dataloader)
                            print(f'Method: {method_name}, seed: {seed}')
                            print(f'\tForward DNNO AUC: {fwd_dnno.mean()}')
                            print(f'\tReverse DNNO AUC: {rev_dnno.mean()}')
                            np.savez(path, fwd_dnno=fwd_dnno, rev_dnno=rev_dnno)
            #supervised_experiment_methods.evaluate_model_performance(model_dir)
            #supervised_experiment_methods.evaluate_leakage_assessments(model_dir, self.load_oracle_assessment())
    
    def run_all_trials(self):
        base_seed = 0
        base_all_kwargs = copy(self.trial_config['default_kwargs'])
        base_all_kwargs.update(self.trial_config['classifiers_pretrain_kwargs'])
        base_all_kwargs.update(self.trial_config['all_kwargs'])
        all_experiment_methods.run_all_hparam_sweep(
            self.all_hparam_sweep_dir, self.profiling_dataset, self.attack_dataset, training_kwargs=base_all_kwargs,
            classifiers_pretrain_trial_count=self.trial_config['all_classifiers_pretrain_htune_trial_count'],
            trial_count=self.trial_config['all_htune_trial_count'],
            max_classifiers_pretrain_steps=self.trial_config['all_classifiers_pretrain_steps'],
            max_steps=self.trial_config['all_train_steps'],
            starting_seed=base_seed,
            reference_leakage_assessment=self.load_oracle_assessment()
        )
        base_seed += self.trial_config['all_classifiers_pretrain_htune_trial_count'] + self.trial_config['all_htune_trial_count']
        selection_dnn_seeds = [int(x.split('=')[1]) for x in os.listdir(self.supervised_selection_dir) if x.split('=')[0] == 'seed']
        selection_dnn_dir = os.path.join(self.supervised_selection_dir, f'seed={selection_dnn_seeds[0]}')
        selection_dnn = supervised_experiment_methods.load_trained_supervised_model(selection_dnn_dir)
        selection_dataloader = self.get_dataloader(split='val', seed=selection_dnn_seeds[0])
        best_all_hparams = all_experiment_methods.get_best_all_hparams(
            self.all_hparam_sweep_dir, self.load_oracle_assessment(), selection_dnn, selection_dataloader
        )
        if self.trial_config['all_classifiers_pretrain_steps'] > 0:
            best_pretrain_hparams = all_experiment_methods.get_best_all_pretrain_hparams(os.path.join(self.all_hparam_sweep_dir, 'classifiers_pretraining'))
        else:
            best_pretrain_hparams = None
        for name, hparams in best_all_hparams.items():
            model_dir = os.path.join(self.all_dir, name)
            print(f'Running experiments for {model_dir} with hparams {hparams}')
            for seed in range(base_seed, base_seed + self.seed_count):
                subdir = os.path.join(model_dir, f'seed={seed}')
                kwargs = copy(base_all_kwargs)
                if best_pretrain_hparams is not None:
                    kwargs.update(best_pretrain_hparams)
                kwargs.update(hparams)
                all_experiment_methods.train_all_model(
                    subdir, self.profiling_dataset, self.attack_dataset, training_kwargs=kwargs,
                    max_steps=self.trial_config['all_train_steps'], seed=seed, reference_leakage_assessment=self.load_oracle_assessment(),
                    pretrain_max_steps=self.trial_config['all_classifiers_pretrain_steps'], pretrain_kwargs=best_pretrain_hparams
                )
        base_seed += self.seed_count
        print('Evaluating the sensitivity of ALL to hyperparameters.')
        for seed in base_seed + self.run_particular_seeds:
            hparams = copy(base_all_kwargs)
            if best_pretrain_hparams is not None:
                kwargs.update(best_pretrain_hparams)
            kwargs.update(best_all_hparams['oracle'])
            all_experiment_methods.evaluate_all_hparam_sensitivity(
                os.path.join(self.all_sensitivity_analysis_dir, f'seed={seed}'), self.profiling_dataset, self.attack_dataset,
                training_kwargs=kwargs, max_steps=self.trial_config['all_train_steps'], seed=seed, reference_leakage_assessment=self.load_oracle_assessment(),
                pretrain_max_steps=self.trial_config['all_classifiers_pretrain_steps'], pretrain_kwargs=best_pretrain_hparams,
                pretrain_classifiers_dir=os.path.join(self.all_dir, 'oracle', f'seed={seed-self.seed_count}', 'classifier_pretraining')
            )
    
    # Returns the hyperparameters which yielded the best classification performance
    def get_optimal_supervised_kwargs(self):
        assert os.path.exists(os.path.join(self.supervised_hparam_sweep_dir, 'results.pickle'))
        with open(os.path.join(self.supervised_hparam_sweep_dir, 'results.pickle'), 'rb') as f:
            results = pickle.load(f)
        best_result_idx = np.argmin(results['min_rank'])
        best_result_dir = os.path.join(self.supervised_hparam_sweep_dir, f'trial_{best_result_idx}')
        with open(os.path.join(best_result_dir, 'hparams.json'), 'r') as f:
            best_hparams = json.load(f)
        kwargs = copy(self.trial_config['default_kwargs'])
        kwargs.update(self.trial_config['supervised_training_kwargs'])
        kwargs.update(best_hparams)
        return kwargs
    
    # Re-runs the trial with the best model hyperparameters for multiple random seeds so we can compute error bars.
    def run_supervised_trials_for_attribution(self):
        optimal_supervised_kwargs = self.get_optimal_supervised_kwargs()
        os.makedirs(self.supervised_attribution_dir, exist_ok=True)
        for seed in range(self.seed_count):
            set_seed(seed)
            supervised_trainer = SupervisedTrainer(
                self.profiling_dataset, self.attack_dataset, default_training_module_kwargs=optimal_supervised_kwargs, reference_leakage_assessment=self.load_oracle_assessment()
            )
            supervised_trainer.run(os.path.join(self.supervised_attribution_dir, f'seed={seed}'), max_steps=self.trial_config['supervised_train_steps'], plot_metrics_over_time=True)

    # Re-runs the trial with the best model hyperparameters for multiple random seeds, for use when running DNN occlusion tests. I want these to be independent to avoid any sort of weird coupling between trials.
    def run_supervised_trials_for_selection(self):
        optimal_supervised_kwargs = self.get_optimal_supervised_kwargs()
        #optimal_supervised_kwargs.update({'lr': 3e-4, 'beta_1': 0.5, 'weight_decay': 1e-2, 'lr_scheduler_name': 'CosineDecayLRSched', 'input_dropout': 0.0, 'hidden_dropout': 0.0, 'output_dropout': 0.0})
        os.makedirs(self.supervised_selection_dir, exist_ok=True)
        for seed in range(self.seed_count):
            set_seed(seed)
            supervised_trainer = SupervisedTrainer(
                self.profiling_dataset, self.attack_dataset, default_training_module_kwargs=optimal_supervised_kwargs, reference_leakage_assessment=None #self.load_oracle_assessment()
            )
            supervised_trainer.run(os.path.join(self.supervised_selection_dir, f'seed={seed}'), max_steps=self.trial_config['supervised_train_steps'])
    
    @staticmethod
    def load_supervised_model(model_dir, checkpoint: Literal['early_stop', 'final'] = 'early_stop'):
        checkpoint_path = os.path.join(model_dir, f'{checkpoint}_checkpoint.ckpt')
        module = SupervisedModule.load_from_checkpoint(checkpoint_path)
        classifier = module.classifier
        classifier.eval()
        classifier.requires_grad_(False)
        return classifier

    def run_supervised_attribution(self):
        if not os.path.exists(os.path.join(self.supervised_attribution_dir, 'leakage_assessments.npz')):
            print('Running supervised attribution.')
            leakage_assessments = {
                method: np.full((self.seed_count, self.profiling_dataset.timesteps_per_trace), np.nan, dtype=np.float32)
                for method in ['gradvis', 'saliency', 'lrp', '1-occlusion', 'inputxgrad'] #, 'occpoi']
            }
            for seed in range(self.seed_count):
                set_seed(seed)
                base_dir = os.path.join(self.supervised_attribution_dir, f'seed={seed}')
                classifier = self.load_supervised_model(base_dir, 'early_stop')
                attributor = NeuralNetAttribution(DataLoader(self.profiling_dataset, batch_size=2048, num_workers=1), classifier, seed=seed, device='cuda' if torch.cuda.is_available() else 'cpu')
                leakage_assessments['gradvis'][seed, :] = attributor.compute_gradvis()
                leakage_assessments['saliency'][seed, :] = attributor.compute_saliency()
                leakage_assessments['lrp'][seed, :] = attributor.compute_lrp()
                leakage_assessments['1-occlusion'][seed, :] = attributor.compute_n_occlusion(1)
                leakage_assessments['inputxgrad'][seed, :] = attributor.compute_inputxgrad()
                #occpoi_computer = OccPOI(
                #    DataLoader(self.attack_dataset, batch_size=2048, num_workers=1), classifier, seed=seed, device='cuda' if torch.cuda.is_available() else 'cpu', dataset_name=self.dataset_name
                #)
                #leakage_assessments['occpoi'][seed, :] = occpoi_computer(extended=False)
            assert all(np.all(np.isfinite(x)) for x in leakage_assessments.values())
            np.savez(os.path.join(self.supervised_attribution_dir, 'leakage_assessments.npz'), **leakage_assessments)
        else:
            leakage_assessments = np.load(os.path.join(self.supervised_attribution_dir, 'leakage_assessments.npz'), allow_pickle=True)
        print('oSNR for neural net attribution-based assessments:')
        for method_name, leakage_assessment in leakage_assessments.items():
            corr = np.array([self.compute_oracle_snr_corr(_leakage_assessment) for _leakage_assessment in leakage_assessment])
            print(f'\t{method_name}: {corr.mean()} +/- {corr.std()}')

    # Random hyperparameter search for ALL with various ablations. Search space is hard-coded in training_modules/adversarial_leakage_localization/trainer.py because I'm lazy.
    def run_all_hparam_sweep(self, ablation: Literal[
        'none', # The method as described in the paper.
        'cooperative', # Instead of training the noise generator to maximize loss -> noisier === leakier, we train it to minimize loss -> less-noisy === leakier.
        'noconditioning', # We don't feed the input mask as an auxiliary argument to the classifier.
        'two_stage', # We solely train the classifier for 50% of the steps, followed by solely training the noise generator for 50% of the steps.
        'attribution', # Two-stage training without conditioning. i.e. we have converted our method into a neural net attribution method.
        'mask_norm_penalty', # We omit the budget constraint and instead penalize the sum of the l1 and l2 norms of the input mask, similar to one of our reviewer's papers.
        'gamma_norm_penalty' # We omit the budget constraint and instead penalize the l1 norm of the erasure probabilities, similar to ENCO.
    ] = 'none'):
        experiment_dir = os.path.join(self.all_hparam_sweep_dir, f'ablation={ablation}')
        os.makedirs(experiment_dir, exist_ok=True)
        pretrain_classifiers = self.trial_config['all_classifiers_pretrain_htune_trial_count'] > 0
        if not os.path.exists(os.path.join(experiment_dir, 'results.pickle')):
            print(f'Running ALL hparam sweep with ablation: {ablation}')
            kwargs = copy(self.trial_config['default_kwargs'])
            kwargs.update(self.trial_config['classifiers_pretrain_kwargs'])
            kwargs.update(self.trial_config['all_kwargs'])
            if ablation == 'none':
                classifier_pretrain_hparam_sweep_dir = os.path.join(self.all_hparam_sweep_dir, 'ablation=none', 'pretrained_classifiers')
            elif ablation == 'cooperative':
                kwargs.update({'adversarial_mode': False})
                classifier_pretrain_hparam_sweep_dir = os.path.join(self.all_hparam_sweep_dir, 'ablation=none', 'pretrained_classifiers')
            elif ablation == 'noconditioning':
                kwargs.update({'omit_classifier_conditioning': True})
                classifier_pretrain_hparam_sweep_dir = os.path.join(self.all_hparam_sweep_dir, 'ablation=noconditioning', 'pretrained_classifiers')
            elif ablation == 'two_stage':
                pretrain_classifiers = False
            elif ablation == 'attribution':
                pretrain_classifiers = False
            elif ablation == 'mask_norm_penalty':
                kwargs.update({'penalty_style': 'mask_norm_penalty'})
                classifier_pretrain_hparam_sweep_dir = os.path.join(self.all_hparam_sweep_dir, 'ablation=none', 'pretrained_classifiers')
            elif ablation == 'gamma_norm_penalty':
                kwargs.update({'penalty_style': 'gamma_norm_penalty'})
                classifier_pretrain_hparam_sweep_dir = os.path.join(self.all_hparam_sweep_dir, 'ablation=none', 'pretrained_classifiers')
            else:
                assert False
            all_trainer = ALLTrainer(
                self.profiling_dataset, self.attack_dataset, default_training_module_kwargs=kwargs, reference_leakage_assessment=self.load_oracle_assessment()
            )
            if pretrain_classifiers:
                os.makedirs(classifier_pretrain_hparam_sweep_dir, exist_ok=True)
                best_all_classifier_pretrain_dir = all_trainer.htune_pretrain_classifiers(
                    classifier_pretrain_hparam_sweep_dir,
                    trial_count=self.trial_config['all_classifiers_pretrain_htune_trial_count'],
                    max_steps=self.trial_config['all_classifiers_pretrain_steps'],
                    starting_seed=5
                )
            else:
                best_all_classifier_pretrain_dir = None
            all_trainer.htune_leakage_localization(
                experiment_dir,
                pretrained_classifiers_logging_dir=best_all_classifier_pretrain_dir,
                trial_count=self.trial_config['all_htune_trial_count'],
                max_steps=self.trial_config['all_train_steps'],
                ablation=ablation,
                starting_seed=5 + self.trial_config['all_classifiers_pretrain_htune_trial_count']
            )
    
    def find_best_all_classifiers_pretrain_dir(self):
        base_dir = os.path.join(self.all_hparam_sweep_dir, 'ablation=none', 'pretrained_classifiers')
        if os.path.exists(os.path.join(base_dir, 'results.pickle')):
            with open(os.path.join(base_dir, 'results.pickle'), 'rb') as f:
                results = pickle.load(f)
            best_idx = np.argmin(results['min_rank'])
            dir = os.path.join(base_dir, f'trial_{best_idx}')
            return dir
        else:
            return None
    
    def get_best_model(self, sweep_dir, oracle=False, get_assessment_fn=None, outfile=None, ret_traces=False):
        experiment_dir = os.path.join(sweep_dir)
        if outfile is None:
            outfile = 'selection_criteria.npz'
        if not os.path.exists(os.path.join(experiment_dir, outfile)):
            if get_assessment_fn is None:
                get_assessment_fn = lambda x: np.load(os.path.join(x, 'leakage_assessment.npy'))
            trial_indices = [int(x.split('_')[1]) for x in os.listdir(experiment_dir) if x.split('_')[0] == 'trial']
            trial_count = max(trial_indices)
            oracle_snr_vals = []
            fwd_dnn_occlusion_vals = []
            rev_dnn_occlusion_vals = []
            mean_assessment_corrs = []
            mean_assessment = np.zeros((self.profiling_dataset.timesteps_per_trace,))
            for trial_idx in range(trial_count):
                mean_assessment += get_assessment_fn(os.path.join(experiment_dir, f'trial_{trial_idx}'))
            mean_assessment /= trial_count
            for trial_idx in tqdm(range(trial_count)):
                assessment = get_assessment_fn(os.path.join(experiment_dir, f'trial_{trial_idx}'))
                oracle_snr_vals.append(self.compute_oracle_snr_corr(assessment))
                fwd_dnn_occlusion, rev_dnn_occlusion = self.compute_dnno_auc(assessment, seed=0)
                mean_assessment_corr = spearmanr(assessment, mean_assessment).statistic
                if np.isnan(mean_assessment_corr):
                    mean_assessment_corr = 0.
                fwd_dnn_occlusion_vals.append(fwd_dnn_occlusion)
                rev_dnn_occlusion_vals.append(rev_dnn_occlusion)
                mean_assessment_corrs.append(mean_assessment_corr)
            oracle_snr_vals = np.array(oracle_snr_vals)
            fwd_dnn_occlusion_vals = np.array(fwd_dnn_occlusion_vals)
            rev_dnn_occlusion_vals = np.array(rev_dnn_occlusion_vals)
            mean_assessment_corrs = np.array(mean_assessment_corrs)
            composite_assessment = fwd_dnn_occlusion_vals.argsort().argsort() + (-rev_dnn_occlusion_vals).argsort().argsort() + (-mean_assessment_corrs).argsort().argsort()
            np.savez(
                os.path.join(experiment_dir, outfile), oracle_snr_vals=oracle_snr_vals,
                fwd_dnn_occlusion_vals=fwd_dnn_occlusion_vals, rev_dnn_occlusion_vals=rev_dnn_occlusion_vals,
                mean_assessment_corrs=mean_assessment_corrs, composite_assessment=composite_assessment
            )
        else:
            rv = np.load(os.path.join(experiment_dir, outfile), allow_pickle=True)
            oracle_snr_vals = rv['oracle_snr_vals']
            fwd_dnn_occlusion_vals = rv['fwd_dnn_occlusion_vals']
            rev_dnn_occlusion_vals = rv['rev_dnn_occlusion_vals']
            mean_assessment_corrs = rv['mean_assessment_corrs']
            composite_assessment = rv['composite_assessment']
        fig, axes = plt.subplots(1, 5, figsize=(5*PLOT_WIDTH, PLOT_WIDTH))
        axes[0].plot(oracle_snr_vals, fwd_dnn_occlusion_vals, marker='.', linestyle='none', color='blue', **PLOT_KWARGS)
        axes[1].plot(oracle_snr_vals, rev_dnn_occlusion_vals, marker='.', linestyle='none', color='blue', **PLOT_KWARGS)
        axes[2].plot(oracle_snr_vals, rev_dnn_occlusion_vals - fwd_dnn_occlusion_vals, marker='.', linestyle='none', color='blue', **PLOT_KWARGS)
        axes[3].plot(oracle_snr_vals, mean_assessment_corrs, marker='.', linestyle='none', color='blue', **PLOT_KWARGS)
        axes[4].plot(oracle_snr_vals, composite_assessment, color='blue', marker='.', linestyle='none', **PLOT_KWARGS)
        fig.tight_layout()
        fig.savefig(os.path.join(experiment_dir, 'selection_strategies_vs_oracle.png'), **SAVEFIG_KWARGS)
        plt.close(fig)
        rv = [np.argmax(oracle_snr_vals) if oracle else np.argmin(composite_assessment)]
        if ret_traces:
            rv.extend([oracle_snr_vals, composite_assessment])
        return rv[0] if len(rv) == 1 else tuple(rv)
    
    def find_best_all_hparams(self):
        chosen_idx = self.get_best_model(sweep_dir=os.path.join(self.all_hparam_sweep_dir, r'ablation=none'))
        trial_dir = os.path.join(self.all_hparam_sweep_dir, r'ablation=none', f'trial_{chosen_idx}')
        with open(os.path.join(trial_dir, 'hparams.json'), 'r') as f:
            hparams = json.load(f)
        return hparams

    def _run_all_trials(self):
        optimal_all_hparams = self.find_best_all_hparams()
        pretrain_classifiers = self.trial_config['all_classifiers_pretrain_htune_trial_count'] > 0
        for seed in range(self.seed_count):
            experiment_dir = os.path.join(self.all_dir, f'all_seed={seed}')
            kwargs = copy(self.trial_config['default_kwargs'])
            kwargs.update(self.trial_config['classifiers_pretrain_kwargs'])
            kwargs.update(self.trial_config['all_kwargs'])
            kwargs.update(optimal_all_hparams)
            all_trainer = ALLTrainer(
                self.profiling_dataset, self.attack_dataset, default_training_module_kwargs=kwargs, reference_leakage_assessment=self.load_oracle_assessment()
            )
            if pretrain_classifiers:
                classifier_pretrain_dir = os.path.join(self.all_dir, f'classifiers_pretrain_seed={seed}')
                all_trainer.pretrain_classifiers(classifier_pretrain_dir, max_steps=self.trial_config['all_classifiers_pretrain_steps'])
            else:
                classifier_pretrain_dir = None
            all_trainer.run(
                experiment_dir, pretrained_classifiers_logging_dir=classifier_pretrain_dir,
                max_steps=self.trial_config['all_train_steps'],
                reference=self.load_oracle_assessment()
            )
    
    def plot_model_selection_efficacy(self):
        chosen_indices, oracles, composites = {}, {}, {}
        chosen_indices['all'], oracles['all'], composites['all'] = self.get_best_model(sweep_dir=os.path.join(self.all_hparam_sweep_dir, r'ablation=none'), ret_traces=True)
        for nn_attr_method in ['saliency', 'lrp', 'inputxgrad', 'gradvis']:
            chosen_indices[nn_attr_method], oracles[nn_attr_method], composites[nn_attr_method] = self.get_best_model(
                sweep_dir=self.supervised_hparam_sweep_dir,
                get_assessment_fn=lambda x: np.load(os.path.join(x, 'early_stop_leakage_assessments.npz'), allow_pickle=True)[nn_attr_method],
                outfile=f'{nn_attr_method}_selection_criteria.npz', ret_traces=True
            )
        fig, ax = plt.subplots(1, 1, figsize=(PLOT_WIDTH, PLOT_WIDTH))
        colors = ['blue', 'green', 'purple', 'orange', 'red']
        for method, color in zip(chosen_indices.keys(), colors):
            ax.plot(oracles[method], composites[method], color=color, marker='.', linestyle='none', **PLOT_KWARGS)
        for method, color in zip(chosen_indices.keys(), colors):
            ax.plot([oracles[method][chosen_indices[method]]], [composites[method][chosen_indices[method]]], color=color, marker='*', linestyle='none', markersize=10, **PLOT_KWARGS)
        ax.set_xlabel(r'Oracle agreement $\uparrow$', fontsize=16)
        ax.set_ylabel(r'Model selection criterion $\downarrow$', fontsize=16)
        fig.tight_layout()
        fig.savefig(os.path.join(self.plots_for_paper_dir, 'model_selection_illustration.pdf'), **SAVEFIG_KWARGS)
        plt.close(fig)

    def run_all_trial(self, logging_dir: str, override_hparams: Optional[dict] = None):
        override_hparams = override_hparams or {}
        kwargs = copy(self.trial_config['default_kwargs'])
        kwargs.update(self.trial_config['classifiers_pretrain_kwargs'])
        kwargs.update(self.trial_config['all_kwargs'])
        trainer = ALLTrainer(self.profiling_dataset, self.attack_dataset, default_training_module_kwargs=kwargs, reference_leakage_assessment=self.load_oracle_assessment())
        if self.trial_config['all_classifiers_pretrain_htune_trial_count'] > 0:
            classifiers_pretrain_dir = self.find_best_all_classifiers_pretrain_dir()
        else:
            classifiers_pretrain_dir = None
        trainer.run(logging_dir, pretrained_classifiers_logging_dir=classifiers_pretrain_dir, max_steps=self.trial_config['all_train_steps'], override_kwargs=override_hparams)
    
    def plot_oracle_corr_over_time(self):
        supervised_training_curves_path = os.path.join(self.supervised_attribution_dir, 'seed=0', 'training_curves.pickle')
        with open(supervised_training_curves_path, 'rb') as f:
            supervised_training_curves = pickle.load(f)
        all_chosen_idx = self.get_best_model(sweep_dir=os.path.join(self.all_hparam_sweep_dir, r'ablation=none'))
        all_training_curves_path = os.path.join(self.all_hparam_sweep_dir, r'ablation=none', f'trial_{all_chosen_idx}', 'training_curves.pickle')
        with open(all_training_curves_path, 'rb') as f:
            all_training_curves = pickle.load(f)
        oracle_corr_traces = {
            'all': all_training_curves['oracle_snr_corr'],
            'gradvis': supervised_training_curves['gradvis_oracle_agreement'],
            'saliency': supervised_training_curves['saliency_oracle_agreement'],
            'lrp': supervised_training_curves['lrp_oracle_agreement'],
            'inputxgrad': supervised_training_curves['inputxgrad_oracle_agreement']
        }
        fig, ax = plt.subplots(1, 1, figsize=(PLOT_WIDTH, PLOT_WIDTH))
        for method_name, (x, y) in oracle_corr_traces.items():
            if method_name == 'all':
                x = np.linspace(0, 2*max(x), 2*len(x))
                y = np.concatenate([np.zeros(len(y)), y])
            ax.plot(x, y, label=method_name, linestyle='-', marker='.', **PLOT_KWARGS)
        ax.set_xlabel('Training step', fontsize=16)
        ax.set_ylabel('Oracle agreement', fontsize=16)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.plots_for_paper_dir, 'oracle_agreement_over_time.pdf'), **SAVEFIG_KWARGS)

    def evaluate_all_hparam_sensitivity(self):
        gamma_bar_vals = np.arange(0.05, 1.0, 0.05)
        theta_lr_scalar_vals = np.logspace(-1, 1, 11)
        etat_lr_scalar_vals = np.logspace(-1, 1, 11)

        for seed in range(self.seed_count):
            for gamma_bar in gamma_bar_vals:
                trial_dir = os.path.join(self.all_sensitivity_analysis_dir, f'seed={seed}', f'gamma_bar={gamma_bar}')
                if not os.path.exists(os.path.join(trial_dir, 'leakage_assessment.npy')):
                    hparams = self.find_best_all_hparams()
                    hparams.update({'gamma_bar': gamma_bar})
                    set_seed(seed)
                    self.run_all_trial(trial_dir, hparams)
            for theta_lr_scalar in theta_lr_scalar_vals:
                for etat_lr_scalar in etat_lr_scalar_vals:
                    if not os.path.exists(os.path.join(trial_dir, 'leakage_assessment.npy')):
                        trial_dir = os.path.join(self.all_sensitivity_analysis_dir, f'seed={seed}', f'theta_lr_sc={theta_lr_scalar}__etat_lr_sc={etat_lr_scalar}')
                        hparams = self.find_best_all_hparams()
                        hparams.update({'theta_lr': theta_lr_scalar*hparams['theta_lr'], 'etat_lr': etat_lr_scalar*hparams['etat_lr']})
                        set_seed(seed)
                        self.run_all_trial(trial_dir, hparams)

        # Plot performance vs. gamma_bar
        os.makedirs(self.plots_for_paper_dir, exist_ok=True)
        optimal_hparams = self.find_best_all_hparams()
        gamma_bar_sweep_perf = np.full((self.seed_count, len(gamma_bar_vals)), np.nan, dtype=np.float32)
        for seed in range(self.seed_count):
            for idx, gamma_bar in enumerate(gamma_bar_vals):
                trial_dir = os.path.join(self.all_sensitivity_analysis_dir, f'seed={0}', f'gamma_bar={gamma_bar}')
                leakage_assessment = np.load(os.path.join(trial_dir, 'leakage_assessment.npy'))
                gamma_bar_sweep_perf[seed, idx] = self.compute_oracle_snr_corr(leakage_assessment)
        assert np.all(np.isfinite(gamma_bar_sweep_perf))
        fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_WIDTH))
        chosen_idx = np.argmin(np.abs(gamma_bar_vals - optimal_hparams['gamma_bar']))
        ax.errorbar(gamma_bar_vals, gamma_bar_sweep_perf.mean(axis=0), gamma_bar_sweep_perf.std(axis=0), fmt='.', markersize=2, color='blue', ecolor='blue', **PLOT_KWARGS)
        ax.plot(gamma_bar_vals[chosen_idx], gamma_bar_sweep_perf.mean(axis=0)[chosen_idx], linestyle='none', color='blue', markersize=10, marker='*', **PLOT_KWARGS)
        ax.axhline(0, color='black', linestyle='--')
        ax.set_xlabel(r'Budget hyperparameter: $\overline{\gamma}$', fontsize=16)
        ax.set_ylabel(r'Agreement with oracle $\uparrow$', fontsize=16)
        fig.tight_layout()
        fig.savefig(os.path.join(self.plots_for_paper_dir, f'{self.dataset_name}_sensitivity_to_gammabar.pdf'), **SAVEFIG_KWARGS)
        plt.close(fig)

        lr_scalar_sweep_perf = np.full((self.seed_count, len(theta_lr_scalar_vals), len(etat_lr_scalar_vals)), np.nan, dtype=np.float32)
        for seed in range(self.seed_count):
            for tidx, theta_lr_scalar in enumerate(theta_lr_scalar_vals):
                for eidx, etat_lr_scalar in enumerate(etat_lr_scalar_vals):
                    trial_dir = os.path.join(self.all_sensitivity_analysis_dir, f'seed={seed}', f'theta_lr_sc={theta_lr_scalar}__etat_lr_sc={etat_lr_scalar}')
                    leakage_assessment = np.load(os.path.join(trial_dir, 'leakage_assessment.npy'))
                    lr_scalar_sweep_perf[seed, tidx, eidx] = self.compute_oracle_snr_corr(leakage_assessment)
        assert np.all(np.isfinite(lr_scalar_sweep_perf))
        fig, ax = plt.subplots(figsize=(1.2*PLOT_WIDTH, PLOT_WIDTH))
        contour = ax.contourf(optimal_hparams['theta_lr']*theta_lr_scalar_vals, optimal_hparams['etat_lr']*etat_lr_scalar_vals, lr_scalar_sweep_perf.mean(axis=0), levels=50)
        for x in contour.collections:
            x.set_rasterized(True)
        ax.plot([optimal_hparams['theta_lr']], [optimal_hparams['etat_lr']], marker='*', markersize=10, linestyle='none', color='blue', **PLOT_KWARGS)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'Learning rate for $\boldsymbol{\theta}$', fontsize=16)
        ax.set_ylabel(r'Learning rate for $\tilde{\boldsymbol{\eta}}$', fontsize=16)
        cbar = plt.colorbar(contour)
        cbar.ax.set_ylabel(r'Agreement with oracle $\uparrow$', fontsize=16)
        fig.tight_layout()
        fig.savefig(os.path.join(self.plots_for_paper_dir, f'{self.dataset_name}_sensitivity_to_lr.pdf'), **SAVEFIG_KWARGS)
        plt.close(fig)
    
    def plot_all_hparam_sweep(self):
        plot_hparam_sweep_results(self.all_hparam_sweep_dir)
    
    def qualitative_oracle_agreement_plot(self):
        best_all_trial_idx = self.get_best_model(sweep_dir=os.path.join(self.all_hparam_sweep_dir, r'ablation=none'), oracle=True)
        best_nn_attr_idx = self.get_best_model(
            sweep_dir=self.supervised_hparam_sweep_dir,
            get_assessment_fn=lambda x: np.load(os.path.join(x, 'early_stop_leakage_assessments.npz'), allow_pickle=True)['gradvis'],
            outfile='gradvis_selection_criteria.npz', oracle=True
        )
        all_assessment = np.load(os.path.join(self.all_hparam_sweep_dir, r'ablation=none', f'trial_{best_all_trial_idx}', 'leakage_assessment.npy'))
        nn_attr_assessment = np.load(os.path.join(self.supervised_hparam_sweep_dir, f'trial_{best_nn_attr_idx}', 'early_stop_leakage_assessments.npz'), allow_pickle=True)['gradvis']
        oracle_assessment = self.load_oracle_assessment()
        fig, ax = plt.subplots(1, 1, figsize=(1.2*PLOT_WIDTH, PLOT_WIDTH))
        tax = ax.twinx()
        all_max = all_assessment.max()
        fwd = lambda x: np.log10(all_max/x)
        rev = lambda x: all_max / 10**x
        tax.set_yscale('function', functions=(fwd, rev))
        tax.set_ylim(rev(-0.1), rev(1.1))
        tax.invert_yaxis()
        ax.plot(oracle_assessment, nn_attr_assessment, linestyle='none', color='red', markersize=3, marker='s', alpha=0.5, **PLOT_KWARGS)
        tax.plot(oracle_assessment, all_assessment, linestyle='none', color='blue', markersize=2, marker='.', **PLOT_KWARGS)
        ax.set_xlabel(r'Oracle leakiness', fontsize=16)
        tax.set_ylabel(r'Leakiness estimate by ALL (ours)', color='blue', fontsize=16)
        ax.set_ylabel(r'Leakiness estimate by GradVis', color='red', fontsize=16)
        ax.set_xscale('log')
        ax.set_yscale('log')
        tax.set_yscale('log')
        fig.tight_layout()
        fig.savefig(os.path.join(self.plots_for_paper_dir, 'qualitative_correlation_with_oracle.pdf'), **SAVEFIG_KWARGS)
        plt.close(fig)
    
    def __call__(self,
        compute_random: bool = False,
        compute_1o_parametric_stats: bool = False,
        run_supervised_trials: bool = False,
        run_all_trials: bool = False,
        run_supervised_hparam_sweep: bool = False,
        do_supervised_training: bool = False,
        run_supervised_attribution: bool = False,
        run_all_hparam_sweep: bool = False,
        run_all: bool = False,
        eval_all_sensitivity: bool = False
    ):
        self.construct_datasets()
        if compute_1o_parametric_stats:
            self.compute_first_order_parametric_stats()
            self.plot_first_order_parametric_assessments()
            self.plot_oracle_assessments()
        if compute_random:
            self.compute_random_assessment()
            self.plot_random_assessment()
        if run_supervised_trials:
            self.run_supervised_trials()
        if run_all_trials:
            self.run_all_trials()
        r"""
        if do_supervised_training:
            self.run_supervised_trials_for_attribution()
            self.run_supervised_trials_for_selection()
        if run_supervised_attribution:
            self.run_supervised_attribution()
        if run_all_hparam_sweep:
            for ablation in ['none']: #, 'cooperative', 'noconditioning', 'mask_norm_penalty', 'gamma_norm_penalty']:
                self.run_all_hparam_sweep(ablation)
            #self.plot_all_hparam_sweep()
        if run_all:
            self.get_best_model(sweep_dir=os.path.join(self.all_hparam_sweep_dir, r'ablation=none'))
            for nn_attr_method in ['saliency', 'lrp', 'inputxgrad', 'gradvis']:
                self.get_best_model(
                    sweep_dir=self.supervised_hparam_sweep_dir,
                    get_assessment_fn=lambda x: np.load(os.path.join(x, 'early_stop_leakage_assessments.npz'), allow_pickle=True)[nn_attr_method],
                    outfile=f'{nn_attr_method}_selection_criteria.npz'
                )
            self.qualitative_oracle_agreement_plot()
            self.plot_model_selection_efficacy()
            self.evaluate_all_hparam_sensitivity()
            self.plot_oracle_corr_over_time()"""