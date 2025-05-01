from typing import Literal, Dict, Any, Optional
import os

import numpy as np
from torch.utils.data import Dataset

from common import *
from utils.baseline_assessments import FirstOrderStatistics

class Trial:
    def __init__(self,
        dataset_name: Literal['ascadv1-fixed', 'ascadv1-variable', 'dpav4', 'aes-hd', 'otiait', 'otp'],
        trial_config: Dict[str, Any],
        seed_count: int = 1,
        logging_dir: Optional[str] = None
    ):
        self.dataset_name = dataset_name
        self.trial_config = trial_config
        self.seed_count = seed_count
        self.logging_dir = logging_dir or dataset_name.replace('-', '_')
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
    
    def compute_random_assessment(self):
        os.makedirs(self.random_assessment_dir, exist_ok=True)
        if not os.path.exists(os.path.join(self.random_assessment_dir, 'assessment.npy')):
            print('Computing random assessment.')
            assessment = np.random.randn(self.seed_count, self.profiling_dataset.timesteps_per_trace)
            np.save(os.path.join(self.random_assessment_dir, 'assessment.npy'), assessment)
        else:
            print('Random assessment already exists.')
            assessment = np.load(os.path.join(self.random_assessment_dir, 'assessment.npy'))
            seed_count, timestep_count = assessment.shape
            assert timestep_count == self.profiling_dataset.timesteps_per_trace
            if seed_count < self.seed_count:
                print(f'\tNot enough seeds. Running {self.seed_count-seed_count} more trials.')
                assessment = np.concatenate([np.random.randn(self.seed_count-seed_count, timestep_count), assessment], axis=0)
                np.save(os.path.join(self.random_assessment_dir, 'assessment.npy'), assessment)
    
    def load_random_assessment(self) -> Optional[np.ndarray]:
        if not os.path.exists(os.path.join(self.random_assessment_dir, 'assessment.npy')):
            return None
        else:
            return np.load(os.path.join(self.random_assessment_dir, 'assessment.npy'))
    
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
                    np.save(path, getattr(stats_computer, f'{method}_vals'))
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
    
    def load_oracle_assessment(self, reduce=False) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        rv = np.zeros(self.profiling_dataset.timesteps_per_trace, dtype=np.float) if reduce else {}
        for target in self.oracle_targets:
            path = os.path.join(self.first_order_parametric_stats_dir, f'attack_snr_{target}.npy')
            assessment = np.load(path)
            if reduce:
                rv += assessment
            else:
                rv[target] = assessment
        return rv
    
    def __call__(self,
        compute_random: bool = False,
        compute_1o_parametric_stats: bool = False
    ):
        self.construct_datasets()
        if compute_random:
            self.compute_random_assessment()
        if compute_1o_parametric_stats:
            self.compute_first_order_parametric_stats()