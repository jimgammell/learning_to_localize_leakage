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
    'otp': 5,
    'nucleo': 3 # FIXME
}

class Trial:
    def __init__(self,
        dataset_name: Literal['ascadv1-fixed', 'ascadv1-variable', 'dpav4', 'aes-hd', 'otiait', 'otp', 'nucleo'],
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
        self.pretrained_model_experiment_dir = os.path.join(self.logging_dir, 'pretrained_model_experiments')
        self.supervised_dropout_ablation = os.path.join(self.logging_dir, 'supervised_dropout_ablation_hparam_sweep')
        self.all_cooperative_ablation_dir = os.path.join(self.logging_dir, 'all_cooperative_ablation')
        self.all_unconditional_ablation_dir = os.path.join(self.logging_dir, 'all_unconditional_ablation')
        self.all_interpretive_ablation_dir = os.path.join(self.logging_dir, 'all_interpretive_ablation')

        if self.dataset_name in ['ascadv1-fixed', 'ascadv1-variable']:
            self.oracle_targets = [ # based on Egger (2021) findings
                'r_in', 'r', 'r_out', 'plaintext__key__r_in', 'subbytes__r', 'subbytes__r_out', 's_prev__subbytes__r_out', 'security_load'
            ]
        elif self.dataset_name in ['dpav4', 'aes-hd', 'otiait', 'otp']: # since these are unprotected, I'm only considering first-order leakage
            self.oracle_targets = ['label']
        elif self.dataset_name in 'nucleo':
            self.oracle_targets = ['r0', 'subbytes__r0', 'r1', 'subbytes__r1']
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
        elif self.dataset_name == 'nucleo':
            from datasets.nucleo import Nucleo
            self.profiling_dataset = Nucleo(root=self.data_dir, train=True)
            self.attack_dataset = Nucleo(root=self.data_dir, train=False)
        else:
            assert False

    def load_oracle_assessment(self, reduce=True) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        rv = {}
        for target in self.oracle_targets:
            path = os.path.join(self.first_order_parametric_stats_dir, f'attack_snr_{target}.npy')
            assessment = np.load(path)
            rv[target] = assessment
        if reduce:
            rv = np.mean(np.stack(list(rv.values())), axis=0)
        return rv
    
    def get_dataloader(self, split: Literal['train', 'val', 'profile', 'attack'] = 'train', seed=0, num_workers=None):
        set_seed(seed)
        data_module = DataModule(self.profiling_dataset, self.attack_dataset, num_workers=num_workers)
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
    
    def run_parametric_trials(self):
        if self.dataset_name == 'nucleo':
            self.profiling_dataset.desync_level = 0
            self.profiling_dataset.configure_target('id')
            self.attack_dataset.configure_target('id')
        set_seed(0)
        print('Running parametric statistics-based methods.')
        os.makedirs(self.first_order_parametric_stats_dir, exist_ok=True)
        for target in self.oracle_targets:
            stats_computor = FirstOrderStatistics(self.attack_dataset, target, chunk_size=1)
            snr = stats_computor.snr_vals[target].reshape(-1)
            np.save(os.path.join(self.first_order_parametric_stats_dir, f'attack_snr_{target}.npy'), snr)
        if not os.path.exists(os.path.join(self.first_order_parametric_stats_dir, 'random.npy')):
            random_assessment = np.random.rand(5, self.profiling_dataset.timesteps_per_trace)
            np.save(os.path.join(self.first_order_parametric_stats_dir, 'random.npy'), random_assessment)
        else:
            random_assessment = np.load(os.path.join(self.first_order_parametric_stats_dir, 'random.npy'))
        if not os.path.exists(os.path.join(self.first_order_parametric_stats_dir, 'first_order_stats.npz')):
            stats_computor = FirstOrderStatistics(self.profiling_dataset, 'label', chunk_size=1)
            snr = stats_computor.snr_vals['label'].reshape(-1)
            sosd = stats_computor.sosd_vals['label'].reshape(-1)
            cpa = stats_computor.cpa_vals['label'].reshape(-1)
            np.savez(os.path.join(self.first_order_parametric_stats_dir, 'first_order_stats.npz'), snr=snr, sosd=sosd, cpa=cpa)
        else:
            rv = np.load(os.path.join(self.first_order_parametric_stats_dir, 'first_order_stats.npz'), allow_pickle=True)
            snr = rv['snr']
            sosd = rv['sosd']
            cpa = rv['cpa']
        self.evaluate_leakage_assessment(snr, dest=os.path.join(self.first_order_parametric_stats_dir, 'snr_evaluation_metrics.npz'), print_res=True)
        self.evaluate_leakage_assessment(sosd, dest=os.path.join(self.first_order_parametric_stats_dir, 'sosd_evaluation_metrics.npz'), print_res=True)
        self.evaluate_leakage_assessment(cpa, dest=os.path.join(self.first_order_parametric_stats_dir, 'cpa_evaluation_metrics.npz'), print_res=True)
        for seed_idx in range(5):
            self.evaluate_leakage_assessment(random_assessment[seed_idx, :], dest=os.path.join(self.first_order_parametric_stats_dir, f'random_evaluation_metrics_{seed_idx}.npz'), print_res=True)
    
    def run_supervised_trials(self):
        if self.dataset_name == 'nucleo':
            self.profiling_dataset.desync_level = 10
            self.profiling_dataset.configure_target('hw')
            self.attack_dataset.configure_target('hw')
        print('Running supervised trials.')
        base_seed = 0
        base_supervised_kwargs = copy(self.trial_config['default_kwargs'])
        base_supervised_kwargs.update(self.trial_config['supervised_training_kwargs'])
        supervised_experiment_methods.run_supervised_hparam_sweep(
            self.supervised_hparam_sweep_dir, self.profiling_dataset, self.attack_dataset, training_kwargs=base_supervised_kwargs,
            trial_count=self.trial_config['supervised_htune_trial_count'], max_steps=self.trial_config['supervised_train_steps'], starting_seed=base_seed
        )
        # FIXME 
        #supervised_experiment_methods.run_supervised_hparam_sweep(
        #    self.supervised_dropout_ablation, self.profiling_dataset, self.attack_dataset, training_kwargs=base_supervised_kwargs, heavy_dropout_ablation=True,
        #    trial_count=self.trial_config['supervised_htune_trial_count'], max_steps=self.trial_config['supervised_train_steps'], starting_seed=base_seed
        #)
        base_seed += self.trial_config['supervised_htune_trial_count']
        best_supervised_hparams = supervised_experiment_methods.get_best_supervised_model_hparams(
            self.supervised_hparam_sweep_dir, self.profiling_dataset, self.attack_dataset, self.dataset_name, self.load_oracle_assessment()
        )
        print('Training selection models...')
        for seed in base_seed + self.run_particular_seeds:
            best_classification_kwargs = copy(base_supervised_kwargs)
            best_classification_kwargs.update(best_supervised_hparams['classification'])
            supervised_experiment_methods.train_supervised_model(
                os.path.join(self.supervised_selection_dir, f'seed={seed}'), self.profiling_dataset, self.attack_dataset, training_kwargs=best_classification_kwargs,
                max_steps=self.trial_config['supervised_train_steps'], seed=seed
            )
        base_seed += 5
        print('Training classification models...')
        for name, hparams in best_supervised_hparams.items():
            if name != 'classification': # FIXME
                continue
            model_dir = os.path.join(self.supervised_attribution_dir, name)
            print(f'Running experiments for {model_dir} with hparams {hparams}')
            for seed in base_seed + self.run_particular_seeds:
                seed_idx = seed - base_seed
                occlusion_windows = np.arange(1, 51, 2) if self.dataset_name in ['dpav4', 'aes-hd'] else np.arange(1, 21, 2)
                subdir = os.path.join(model_dir, f'seed={seed}')
                kwargs = copy(base_supervised_kwargs)
                kwargs.update(hparams)
                supervised_experiment_methods.train_supervised_model(
                    subdir, self.profiling_dataset, self.attack_dataset, training_kwargs=kwargs, dataset_name=self.dataset_name,
                    max_steps=self.trial_config['supervised_train_steps'], seed=seed, reference_leakage_assessment=self.load_oracle_assessment()
                )
                self.evaluate_supervised_model(subdir, seed_idx=seed_idx, print_res=True, cost='reduced')
                # experiments with a handful of publicly-released pretrained models as a sanity check
                if self.dataset_name == 'ascadv1-fixed':
                    self.evaluate_supervised_model(os.path.join(self.pretrained_model_experiment_dir, 'benadjila_cnn_best'), seed_idx=seed_idx, print_res=True)
                    self.evaluate_supervised_model(os.path.join(self.pretrained_model_experiment_dir, 'benadjila_mlp_best'), seed_idx=seed_idx, print_res=True)
                    self.evaluate_supervised_model(os.path.join(self.pretrained_model_experiment_dir, 'wouters', f'seed={seed_idx}'), seed_idx=seed_idx, print_res=True)
                    self.evaluate_supervised_model(os.path.join(self.pretrained_model_experiment_dir, 'zaid', f'seed={seed_idx}'), seed_idx=seed_idx, print_res=True)
                #elif self.dataset_name == 'ascadv1-variable':
                #    self.evaluate_supervised_model(os.path.join(self.pretrained_model_experiment_dir, 'benadjila_cnn_best'), seed_idx=seed_idx, print_res=True)
                elif self.dataset_name == 'aes-hd':
                    self.evaluate_supervised_model(os.path.join(self.pretrained_model_experiment_dir, 'wouters', f'seed={seed_idx}'), seed_idx=seed_idx, print_res=True)
                    self.evaluate_supervised_model(os.path.join(self.pretrained_model_experiment_dir, 'zaid', f'seed={seed_idx}'), seed_idx=seed_idx, print_res=True)
                elif self.dataset_name == 'dpav4':
                    self.evaluate_supervised_model(os.path.join(self.pretrained_model_experiment_dir, 'wouters', f'seed={seed_idx}'), seed_idx=seed_idx, print_res=True)
                    self.evaluate_supervised_model(os.path.join(self.pretrained_model_experiment_dir, 'zaid', f'seed={seed_idx}'), seed_idx=seed_idx, print_res=True)
        print('Doing neural net attribution assessments...')
        for seed_idx in self.run_particular_seeds:
            for x in tqdm((self.trial_config['supervised_htune_trial_count']//5)*seed_idx + np.arange(self.trial_config['supervised_htune_trial_count']//5)):
                model_dir = os.path.join(self.supervised_hparam_sweep_dir, f'trial_{x}')
                print(f'Evaluating model in {model_dir}...')
                self.evaluate_supervised_model(model_dir, seed_idx=0, print_res=True)
                #model_dir = os.path.join(self.supervised_dropout_ablation, f'trial_{x}')
                #print(f'Evaluating model in {model_dir}...')
                #self.evaluate_supervised_model(model_dir, seed_idx=0, print_res=True, skip_metrics=True)
        print('Computing selection criteria...')
        self.compute_selection_criterion_for_attribution_prefix('gradvis')
        self.compute_selection_criterion_for_attribution_prefix('lrp')
        self.compute_selection_criterion_for_attribution_prefix('saliency')
        self.compute_selection_criterion_for_attribution_prefix('1-occlusion')
        self.compute_selection_criterion_for_attribution_prefix('inputxgrad')
        self.compute_selection_criterion_for_attribution_prefix(f'{OPTIMAL_WINDOW_SIZES[self.dataset_name]}-occlusion')
    
    def run_all_hsweep(self, output_dir, base_seed, base_all_kwargs):
        if self.dataset_name == 'nucleo':
            self.profiling_dataset.desync_level = 10
            self.profiling_dataset.configure_target('hw')
            self.attack_dataset.configure_target('hw')
        all_experiment_methods.run_all_hparam_sweep(
            output_dir, self.profiling_dataset, self.attack_dataset, training_kwargs=base_all_kwargs,
            classifiers_pretrain_trial_count=self.trial_config['all_classifiers_pretrain_htune_trial_count'],
            trial_count=self.trial_config['all_htune_trial_count'],
            max_classifiers_pretrain_steps=self.trial_config['all_classifiers_pretrain_steps'],
            max_steps=self.trial_config['all_train_steps'],
            starting_seed=base_seed,
            reference_leakage_assessment=self.load_oracle_assessment()
        )

    def run_all_trials(self):
        if self.dataset_name == 'nucleo':
            self.profiling_dataset.desync_level = 10
            self.profiling_dataset.configure_target('hw')
            self.attack_dataset.configure_target('hw')
        base_seed = 0
        base_all_kwargs = copy(self.trial_config['default_kwargs'])
        base_all_kwargs.update(self.trial_config['classifiers_pretrain_kwargs'])
        base_all_kwargs.update(self.trial_config['all_kwargs'])
        self.run_all_hsweep(self.all_hparam_sweep_dir, 0, base_all_kwargs)
        #cooperative_all_kwargs = copy(base_all_kwargs)
        #cooperative_all_kwargs['adversarial_mode'] = False
        #self.run_all_hsweep(self.all_cooperative_ablation_dir, 0, cooperative_all_kwargs)
        #unconditional_all_kwargs = copy(base_all_kwargs)
        #unconditional_all_kwargs['omit_classifier_conditioning'] = True
        #self.run_all_hsweep(self.all_unconditional_ablation_dir, 0, unconditional_all_kwargs)
        #interpretive_all_kwargs = copy(base_all_kwargs)
        #interpretive_all_kwargs['classifier_to_interpret'] = supervised_experiment_methods.load_trained_supervised_model(os.path.join(self.supervised_attribution_dir, 'classification', 'seed=55'))
        #self.run_all_hsweep(self.all_interpretive_ablation_dir, 0, interpretive_all_kwargs)
        for x in tqdm(range(self.trial_config['all_htune_trial_count'])):
            model_dir = os.path.join(self.all_hparam_sweep_dir, f'trial_{x}')
            leakage_assessment = np.load(os.path.join(model_dir, 'leakage_assessment.npy'))
            self.evaluate_leakage_assessment(leakage_assessment, 0, dest=os.path.join(model_dir, 'evaluation_metrics.npz'))
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
        print(f'Best ALL hparams: {best_all_hparams}')
        print(f'Best ALL pretrain hparams: {best_pretrain_hparams}')
        for name, hparams in best_all_hparams.items():
            model_dir = os.path.join(self.all_dir, name)
            print(f'Running experiments for {model_dir} with hparams {hparams}')
            results = defaultdict(list)
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
                leakage_assessment = np.load(os.path.join(subdir, 'all_training', 'leakage_assessment.npy'))
                _results = self.evaluate_leakage_assessment(leakage_assessment, seed-base_seed, dest=os.path.join(subdir, 'evaluation_metrics.npz'))
                for key, val in _results.items():
                    results[key].append(val)
            print('\tDone.')
            results = {key: np.array(val) for key, val in results.items()}
            for key, val in results.items():
                print(f'\t{key}: {val.mean()} +/- {val.std()}')
        base_seed += 5
        r"""print('Evaluating the sensitivity of ALL to hyperparameters.')
        for seed in base_seed + self.run_particular_seeds:
            hparams = copy(base_all_kwargs)
            if best_pretrain_hparams is not None:
                kwargs.update(best_pretrain_hparams)
            kwargs.update(best_all_hparams['oracle'])
            all_experiment_methods.evaluate_all_hparam_sensitivity(
                os.path.join(self.all_sensitivity_analysis_dir, f'seed={seed}'), self.profiling_dataset, self.attack_dataset,
                training_kwargs=kwargs, max_steps=self.trial_config['all_train_steps'], seed=seed, reference_leakage_assessment=self.load_oracle_assessment(),
                pretrain_max_steps=self.trial_config['all_classifiers_pretrain_steps'], pretrain_kwargs=best_pretrain_hparams,
                pretrain_classifiers_dir=os.path.join(self.all_dir, 'oracle', f'seed={seed-self.seed_count}', 'classifier_pretraining') if self.dataset_name in ['ascadv1-fixed', 'ascadv1-variable', 'aes-hd'] else None
            )"""


    def compute_dnn_occlusion_tests(self, leakage_assessment, seed_idx, quiet: bool = True):
        available_seeds = [int(x.split('=')[1]) for x in os.listdir(self.supervised_selection_dir) if x.split('=')[0] == 'seed']
        available_seeds.sort()
        seed = available_seeds[seed_idx]
        model = supervised_experiment_methods.load_trained_supervised_model(os.path.join(self.supervised_selection_dir, f'seed={seed}'))
        dataloader = self.get_dataloader(split='attack', num_workers=0)
        traces, labels = [], []
        for trace, label in dataloader:
            traces.append(trace)
            labels.append(label)
        traces, labels = torch.cat(traces, dim=0), torch.cat(labels, dim=0)
        if False: #self.dataset_name in ['dpav4', 'aes-hd', 'ascadv1-fixed', 'ascadv1-variable']:
            fwd_dnno = evaluation_methods.run_dnn_occlusion_test(leakage_assessment, model, traces, labels, dataloader=dataloader, performance_metric='traces_to_disclosure', direction='forward', dataset_name=self.dataset_name)
            rev_dnno = evaluation_methods.run_dnn_occlusion_test(leakage_assessment, model, traces, labels, dataloader=dataloader, performance_metric='traces_to_disclosure', direction='reverse', dataset_name=self.dataset_name)
        else:
            fwd_dnno = evaluation_methods.run_dnn_occlusion_test(leakage_assessment, model, traces, labels, direction='forward', quiet=quiet)
            rev_dnno = evaluation_methods.run_dnn_occlusion_test(leakage_assessment, model, traces, labels, direction='reverse', quiet=quiet)
        return fwd_dnno, rev_dnno
    
    def compute_selection_criterion_for_attribution_prefix(self, prefix):
        base_dir = os.path.join(OUTPUT_DIR, self.dataset_name.replace('-', '_'), 'supervised_hparam_sweep')
        mean_assessment = np.zeros(self.profiling_dataset.timesteps_per_trace)
        for trial_idx in range(self.trial_config['supervised_htune_trial_count']):
            assessment = np.load(os.path.join(base_dir, f'trial_{trial_idx}', f'{prefix}.npz'), allow_pickle=True)['attribution']
            mean_assessment += assessment
        mean_assessment /= self.trial_config['supervised_htune_trial_count']
        for trial_idx in range(self.trial_config['supervised_htune_trial_count']):
            assessment = np.load(os.path.join(base_dir, f'trial_{trial_idx}', f'{prefix}.npz'), allow_pickle=True)['attribution']
            self.compute_selection_criterion(assessment, mean_assessment, os.path.join(base_dir, f'trial_{trial_idx}', f'{prefix}_selection_criterion.npz'))

    def compute_selection_criterion(self, leakage_assessment, mean_leakage_assessment, dest):
        if os.path.exists(dest):
            return
        selection_dataloader = self.get_dataloader(split='val', seed=50)
        selection_dnn_dir = os.path.join(self.supervised_selection_dir, 'seed=50')
        selection_dnn = supervised_experiment_methods.load_trained_supervised_model(selection_dnn_dir)
        metrics = {}
        metrics['oracle_agreement'] = evaluation_methods.get_oracle_agreement(leakage_assessment, self.load_oracle_assessment())
        metrics['fwd_dnno_criterion'] = evaluation_methods.get_forward_dnno_criterion(leakage_assessment, selection_dnn, selection_dataloader)
        metrics['rev_dnno_criterion'] = evaluation_methods.get_reverse_dnno_criterion(leakage_assessment, selection_dnn, selection_dataloader)
        if leakage_assessment.std() > 0:
            metrics['mean_agreement'] = spearmanr(leakage_assessment, mean_leakage_assessment).statistic
        else:
            metrics['mean_agreement'] = 0.
        np.savez(dest, **metrics)

    def evaluate_leakage_assessment(self, leakage_assessment, seed_idx=0, dest=None, print_res=False):
        if dest is not None and os.path.exists(dest):
            evaluations = np.load(dest, allow_pickle=True)
            fwd_dnno = evaluations['fwd_dnno']
            rev_dnno = evaluations['rev_dnno']
            ta_ttd = evaluations['ta_ttd']
            oracle_agreement = evaluations['oracle_agreement']
        else:
            fwd_dnno, rev_dnno = self.compute_dnn_occlusion_tests(leakage_assessment, seed_idx, quiet=not print_res)
            ta_ttd = evaluation_methods.run_template_attack_test(leakage_assessment, self.profiling_dataset, self.attack_dataset, dataset_name=self.dataset_name)
            if leakage_assessment.std() == 0:
                oracle_agreement = 0.
            else:
                oracle_assessment = self.load_oracle_assessment()
                oracle_agreement = spearmanr(leakage_assessment, oracle_assessment).statistic
            if dest is not None:
                np.savez(dest, fwd_dnno=fwd_dnno, rev_dnno=rev_dnno, ta_ttd=ta_ttd, oracle_agreement=oracle_agreement)
        if print_res:
            print(f'Evaluation metrics saved to `{dest}`:')
            print(f'\tOracle agreement: {oracle_agreement}')
            print(f'\tTemplate attack MTTD: {ta_ttd}')
            print(f'\tForward DNN occlusion AUC: {fwd_dnno.mean()}')
            print(f'\tReverse DNN occlusion AUC: {rev_dnno.mean()}')
        return dict(fwd_dnno=fwd_dnno, rev_dnno=rev_dnno, ta_ttd=ta_ttd, oracle_agreement=oracle_agreement)
    
    def evaluate_supervised_model(self, output_dir, seed_idx=0, cost: Literal['reduced', 'full'] = 'reduced', print_res=False, skip_metrics=False):
        compute_lrp = False
        standardize_dataset = True
        if 'benadjila_cnn_best' in output_dir:
            if self.dataset_name == 'ascadv1-fixed':
                model_dir = os.path.join(RESOURCE_DIR, 'ascadv1-fixed', 'ASCAD_data', 'ASCAD_trained_models', 'cnn_best_ascad_desync0_epochs75_classes256_batchsize200.h5')
            elif self.dataset_name == 'ascadv1-variable':
                model_dir = os.path.join(RESOURCE_DIR, 'ascadv1-variable', 'cnn2-ascad-desync0.h5')
            else:
                assert False
            standardize_dataset = False
        elif 'benadjila_mlp_best' in output_dir:
            if self.dataset_name == 'ascadv1-fixed':
                model_dir = os.path.join(RESOURCE_DIR, 'ascadv1-fixed', 'ASCAD_data', 'ASCAD_trained_models', 'mlp_best_ascad_desync0_node200_layernb6_epochs200_classes256_batchsize100.h5')
            else:
                assert False
            standardize_dataset = False
        elif 'zaid' in output_dir:
            assert 'seed' in output_dir
            seed = int(output_dir.split(os.sep)[-1].split('=')[1])
            if self.dataset_name == 'ascadv1-fixed':
                model_dir = os.path.join(RESOURCE_DIR, 'zaid_wouters_pretrained', 'pretrained_models', 'models', f'zaid_ascad_desync_0_feature_standardization_{seed}.h5')
            elif self.dataset_name == 'dpav4':
                model_dir = os.path.join(RESOURCE_DIR, 'zaid_wouters_pretrained', 'pretrained_models', 'models', f'zaid_dpav4_feature_standardization_{seed}.h5')
            elif self.dataset_name == 'aes-hd':
                model_dir = os.path.join(RESOURCE_DIR, 'zaid_wouters_pretrained', 'pretrained_models', 'models', f'zaid_aes_hd_feature_standardization_{seed}.h5')
            else:
                assert False
        elif 'wouters' in output_dir:
            assert 'seed' in output_dir
            seed = int(output_dir.split(os.sep)[-1].split('=')[1])
            if self.dataset_name == 'ascadv1-fixed':
                model_dir = os.path.join(RESOURCE_DIR, 'zaid_wouters_pretrained', 'pretrained_models', 'models', f'noConv1_ascad_desync_0_feature_standardization_{seed}.h5')
            elif self.dataset_name == 'dpav4':
                model_dir = os.path.join(RESOURCE_DIR, 'zaid_wouters_pretrained', 'pretrained_models', 'models', f'noConv1_dpav4_feature_standardization_{seed}.h5')
            elif self.dataset_name == 'aes-hd':
                model_dir = os.path.join(RESOURCE_DIR, 'zaid_wouters_pretrained', 'pretrained_models', 'models', f'noConv1_aes_hd_feature_standardization_{seed}.h5')
            else:
                assert False
        else:
            model_dir = output_dir
            compute_lrp = True if self.dataset_name not in ['nucleo'] else False
        if not skip_metrics:
            supervised_experiment_methods.eval_on_attack_dataset(model_dir, self.profiling_dataset, self.attack_dataset, self.dataset_name, output_dir)
        supervised_experiment_methods.attribute_neural_net(
            model_dir, self.profiling_dataset, self.attack_dataset, self.dataset_name, compute_gradvis=True, compute_saliency=True, compute_inputxgrad=True,
            compute_lrp=compute_lrp, compute_occlusion=[1, OPTIMAL_WINDOW_SIZES[self.dataset_name]], output_dir=output_dir
        )
        methods_to_eval = ['gradvis', 'inputxgrad', 'saliency', '1-occlusion', f'{OPTIMAL_WINDOW_SIZES[self.dataset_name]}-occlusion']
        if compute_lrp:
            methods_to_eval.append('lrp')
        if cost == 'full':
            supervised_experiment_methods.attribute_neural_net(
                model_dir, self.profiling_dataset, self.attack_dataset, self.dataset_name, compute_second_order_occlusion=[1, OPTIMAL_WINDOW_SIZES[self.dataset_name]], compute_occpoi=True
            )
            methods_to_eval.extend(['1-second-order-occlusion', f'{OPTIMAL_WINDOW_SIZES[self.dataset_name]}-second-order-occlusion', 'occpoi'])
        if not skip_metrics:
            for method_name in methods_to_eval:
                leakage_assessment = np.load(os.path.join(output_dir, f'{method_name}.npz'), allow_pickle=True)['attribution']
                self.evaluate_leakage_assessment(leakage_assessment, seed_idx, os.path.join(output_dir, f'{method_name}_evaluation_metrics.npz'), print_res=print_res)
    
    def __call__(self,
        run_parametric_trials: bool = False,
        run_supervised_trials: bool = False,
        run_all_trials: bool = False
    ):
        self.construct_datasets()
        if run_parametric_trials:
            self.run_parametric_trials()
        if run_supervised_trials:
            self.run_supervised_trials()
        if run_all_trials:
            self.run_all_trials()