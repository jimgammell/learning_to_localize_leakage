from typing import *
import os
from collections import defaultdict
from copy import copy
from matplotlib import pyplot as plt
from matplotlib.ticker import LinearLocator, LogLocator
import numpy as np
from torch.utils.data import DataLoader

from leakage_localization.common import *
from leakage_localization.datasets.simple_gaussian import SimpleGaussianDataset
from leakage_localization.training_modules.adversarial_leakage_localization import ALLTrainer
from leakage_localization.training_modules.supervised_deep_sca import SupervisedTrainer
from leakage_localization.utils.baseline_assessments import NeuralNetAttribution, FirstOrderStatistics, OccPOI
from leakage_localization.trials.utils import *

to_names = {
    'snr': 'SNR',
    'sosd': 'SoSD',
    'cpa': 'CPA',
    'gradvis': 'GradVis',
    'lrp': 'LRP',
    'saliency': 'saliency',
    'occlusion': 'occlusion',
    'inputxgrad': 'inpXgrad',
    'leakage_localization': 'ALL (Ours)'
}

class Trial:
    def __init__(self,
        logging_dir: Union[str, os.PathLike] = None,
        seed_count: int = 1,
        trial_count: int = 11,
        run_baselines: bool = True
    ):
        self.logging_dir = logging_dir
        self.seed_count = seed_count
        self.trial_count = trial_count
        self.run_kwargs = {'max_steps': 5000, 'anim_gammas': False}
        self.supervised_kwargs = {
            'classifier_name': 'mlp-1d', 'classifier_kwargs': {'layer_count': 1, 'input_dropout': 0.0, 'hidden_dropout': 0.0, 'output_dropout': 0.0},
            'lr': 1.e-4
        }
        self.leakage_localization_kwargs = {
            'classifiers_name': 'mlp-1d', 'classifiers_kwargs': {'layer_count': 1, 'input_dropout': 0.0, 'hidden_dropout': 0.0, 'output_dropout': 0.0},
            'theta_lr': 1.e-4, 'etat_lr': 1e-3, 'adversarial_mode': True, 'gamma_bar': 0.5,
        }
        self.run_baselines = run_baselines
    
    def run_experiments(self, logging_dir, max_leaky_pair_count: int = 13, run_baselines: bool = True):
        leakage_assessments = {key: defaultdict(list) for key in ['gradvis', 'saliency', 'lrp', 'inputxgrad', '1-occlusion', 'snr', 'sosd', 'cpa', 'all']}
        for seed in range(self.seed_count):
            for leaky_pair_count in list(range(max_leaky_pair_count+1))[::-1]:
                print(f'Running trial {leaky_pair_count}...')
                trial_dir = os.path.join(logging_dir, f'seed={seed}', f'leaky_pair_count={leaky_pair_count}', 'parametric')
                os.makedirs(trial_dir, exist_ok=True)
                if not os.path.exists(os.path.join(trial_dir, 'leakage_assessments.npz')):
                    set_seed(seed)
                    profiling_dataset = SimpleGaussianDataset(
                        random_count=1, first_order_count=0, second_order_pair_count=2**leaky_pair_count, repeat_dataset=True, buffer_size=10000
                    )
                    attack_dataset = SimpleGaussianDataset(
                        random_count=1, first_order_count=0, second_order_pair_count=2**leaky_pair_count, repeat_dataset=True, buffer_size=10000
                    )
                    first_order_stats = FirstOrderStatistics(profiling_dataset, chunk_size=1)
                    param_assessments = {
                        'snr': first_order_stats.snr_vals['label'].reshape(-1),
                        'sosd': first_order_stats.sosd_vals['label'].reshape(-1),
                        'cpa': first_order_stats.cpa_vals['label'].reshape(-1)
                    }
                    np.savez(os.path.join(trial_dir, 'leakage_assessment.npz'), **param_assessments)
                else:
                    param_assessments = np.load(os.path.join(trial_dir, 'leakage_assessments.npz'), allow_pickle=True)
                for k, v in param_assessments.items():
                    leakage_assessments[k][leaky_pair_count].append(v)
                trial_dir = os.path.join(logging_dir, f'seed={seed}', f'leaky_pair_count={leaky_pair_count}', 'advll')
                if not os.path.exists(os.path.join(trial_dir, 'leakage_assessment.npy')):
                    set_seed(seed)
                    profiling_dataset = SimpleGaussianDataset(
                        random_count=1, first_order_count=0, second_order_pair_count=2**leaky_pair_count, repeat_dataset=True, buffer_size=10000
                    )
                    attack_dataset = SimpleGaussianDataset(
                        random_count=1, first_order_count=0, second_order_pair_count=2**leaky_pair_count, repeat_dataset=True, buffer_size=10000
                    )
                    self.leakage_localization_kwargs.update({'gamma_bar': 1 - 0.5*2**(-0.1*leaky_pair_count)})
                    all_trainer = ALLTrainer(
                        profiling_dataset, attack_dataset, 
                        default_data_module_kwargs={'train_batch_size': int(0.08*len(profiling_dataset)), 'data_mean': np.array([0.0]), 'data_var': np.array([1.0]), 'num_workers': 0},
                        default_training_module_kwargs={**self.leakage_localization_kwargs}
                    )
                    all_assessment = all_trainer.run(
                        logging_dir=trial_dir,
                        max_steps=self.run_kwargs['max_steps']
                    )
                    leakage_assessment = np.load(os.path.join(trial_dir, 'leakage_assessment.npy'))
                    fig, ax = plt.subplots()
                    ax.plot(leakage_assessment.flatten())
                    fig.savefig(os.path.join(trial_dir, 'res.png'))
                    plt.close(fig)
                else:
                    all_assessment = np.load(os.path.join(trial_dir, 'leakage_assessment.npy'))
                leakage_assessments['all'][leaky_pair_count].append(all_assessment)
                trial_dir = os.path.join(logging_dir, f'seed={seed}', f'leaky_pair_count={leaky_pair_count}', 'supervised')
                if not os.path.exists(os.path.join(trial_dir, 'leakage_assessments.npz')):
                    set_seed(seed)
                    profiling_dataset = SimpleGaussianDataset(
                        random_count=1, first_order_count=0, second_order_pair_count=2**leaky_pair_count, repeat_dataset=True, buffer_size=10000
                    )
                    attack_dataset = SimpleGaussianDataset(
                        random_count=1, first_order_count=0, second_order_pair_count=2**leaky_pair_count, repeat_dataset=True, buffer_size=10000
                    )
                    supervised_trainer = SupervisedTrainer(
                        profiling_dataset, attack_dataset,
                        default_data_module_kwargs={'train_batch_size': int(0.08*len(profiling_dataset)), 'data_mean': np.array([0.0]), 'data_var': np.array([1.0]), 'num_workers': 0},
                        default_training_module_kwargs={**self.supervised_kwargs},
                        dataset_name='toy_gaussian'
                    )
                    supervised_trainer.run(
                        logging_dir=trial_dir,
                        max_steps=self.run_kwargs['max_steps'],
                        compute_leakage_assessments=True,
                        compute_occpoi=False
                    )
                attr_leakage_assessments = np.load(os.path.join(trial_dir, 'early_stop_leakage_assessments.npz'), allow_pickle=True)
                for key, val in attr_leakage_assessments.items():
                    leakage_assessments[key][leaky_pair_count].append(val)
        for k1 in leakage_assessments:
            for k2 in leakage_assessments[k1]:
                leakage_assessments[k1][k2] = np.stack(leakage_assessments[k1][k2])
        np.savez(os.path.join(logging_dir, 'leakage_assessments.npz'), **leakage_assessments)
        return leakage_assessments
    
    def plot_experiments(self, dest, leakage_assessments):
        to_kwargs = {
            'snr': {'color': 'green', 'label': 'SNR', 'markersize': 3, 'marker': 'o', 'linestyle': 'solid'},
            'sosd': {'color': 'green', 'label': 'SOSD', 'markersize': 4, 'marker': '^', 'linestyle': 'dotted'},
            'cpa': {'color': 'green', 'label': 'CPA', 'markersize': 5, 'marker': 's', 'linestyle': 'dashed'},
            'gradvis': {'color': 'purple', 'label': 'GradVis', 'markersize': 3, 'marker': 'D', 'linestyle': 'solid'},
            'saliency': {'color': 'purple', 'label': 'Saliency', 'markersize': 4, 'marker': 'v', 'linestyle': 'dotted'},
            'inputxgrad': {'color': 'purple', 'label': r'Input $*$ grad', 'markersize': 5, 'marker': 'P', 'linestyle': 'dashed'},
            '1-occlusion': {'color': 'purple', 'label': '1-occlusion', 'markersize': 6, 'marker': 'X', 'linestyle': 'dashdot'},
            'lrp': {'color': 'purple', 'label': 'LRP', 'markersize': 7, 'marker': 'o', 'linestyle': 'dotted'},
            'all': {'color': 'blue', 'label': 'ALL (ours)', 'markersize': 7, 'marker': 'o', 'linestyle': 'solid'}
        }
        fig, ax = plt.subplots(1, 1, figsize=(PLOT_WIDTH, PLOT_WIDTH))
        ax.axhline(0.5, linestyle='--', color='red', label='Random')
        ax.axhline(0.0, linestyle='--', color='black', label='Oracle')
        for method, leakage_assessments_for_method in leakage_assessments.items():
            leaky_point_counts, negative_ranks = [], []
            for leaky_pt_cnt, assessment in leakage_assessments_for_method.items():
                assessment = assessment.reshape(-1)
                negative_rank = np.sum(assessment < assessment[0]) / len(assessment)
                leaky_point_counts.append(leaky_pt_cnt)
                negative_ranks.append(negative_rank)
            leaky_point_counts = 2**np.array(leaky_point_counts)
            negative_ranks = np.array(negative_ranks)
            ax.plot(
                leaky_point_counts[:len(negative_ranks)], negative_ranks, alpha=0.5 if method != 'all' else 1.0,
                markeredgecolor='k', linewidth=0.5, **to_kwargs[method], **PLOT_KWARGS)
        ax.set_xscale('log')
        ax.set_xlabel(r'Number of second-order leaky measurements: $D$')
        ax.set_ylabel(r'Rank of nonleaky measurement $\downarrow$')
        ax.legend(ncol=3, loc='upper left', fontsize='x-small')
        fig.tight_layout()
        fig.savefig(dest, **SAVEFIG_KWARGS)
    
    def __call__(self):
        out = self.run_experiments(self.logging_dir,)
        self.plot_experiments(os.path.join(self.logging_dir, 'leakage_assessments.pdf'), out)