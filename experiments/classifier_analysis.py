import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scipy.stats import spearmanr
import torch
from torch import nn

from common import *
from training_modules.supervised_deep_sca import SupervisedModule
from training_modules.adversarial_leakage_localization import ALLModule
from trials.real_trials.supervised_experiment_methods import eval_on_attack_dataset, evaluate_model_performance, attribute_neural_net, evaluate_leakage_assessments

TRIAL_DIR = os.path.join(OUTPUT_DIR, 'classifier_analysis_for_rebuttal')
os.makedirs(TRIAL_DIR, exist_ok=True)

class WrapALLClassifier(nn.Module):
    def __init__(self, all_classifier: nn.Module):
        super().__init__()
        self.all_classifier = all_classifier
        self.input_shape = self.all_classifier.input_shape
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.all_classifier(x, torch.ones_like(x))

for dataset_name in ['ascadv1-fixed', 'ascadv1-variable', 'dpav4', 'aes-hd', 'otiait', 'otp']:
    results_dir = os.path.join(OUTPUT_DIR, dataset_name.replace('-', '_'))
    data_dir = os.path.join(RESOURCE_DIR, dataset_name)
    output_dir = os.path.join(TRIAL_DIR, dataset_name)
    supervised_model_checkpoints = [
        os.path.join(results_dir, 'supervised_models_for_attribution', 'classification', f'seed={seed}', 'early_stop_checkpoint.ckpt')
        for seed in [55, 56, 57, 58, 59]
    ]
    all_checkpoints = [
        os.path.join(results_dir, 'all_runs', 'fair', f'seed={seed}', 'all_training', 'final_checkpoint.ckpt')
        for seed in [50, 51, 52, 53, 54]
    ]
    if dataset_name == 'ascadv1-fixed':
        from datasets.ascadv1 import ASCADv1
        profiling_dataset = ASCADv1(root=data_dir, train=True, variable_keys=False)
        attack_dataset = ASCADv1(root=data_dir, train=False, variable_keys=False)
        oracle_targets = [ # based on Egger (2021) findings
                'r_in', 'r', 'r_out', 'plaintext__key__r_in', 'subbytes__r', 'subbytes__r_out', 's_prev__subbytes__r_out', 'security_load'
        ]
    elif dataset_name == 'ascadv1-variable':
        from datasets.ascadv1 import ASCADv1
        profiling_dataset = ASCADv1(root=data_dir, train=True, variable_keys=True)
        attack_dataset = ASCADv1(root=data_dir, train=False, variable_keys=True)
        oracle_targets = [ # based on Egger (2021) findings
                'r_in', 'r', 'r_out', 'plaintext__key__r_in', 'subbytes__r', 'subbytes__r_out', 's_prev__subbytes__r_out', 'security_load'
        ]
    elif dataset_name == 'dpav4':
        from datasets.dpav4 import DPAv4
        profiling_dataset = DPAv4(root=data_dir, train=True)
        attack_dataset = DPAv4(root=data_dir, train=False)
        oracle_targets = ['label']
    elif dataset_name == 'aes-hd':
        from datasets.aes_hd import AES_HD
        profiling_dataset = AES_HD(root=data_dir, train=True)
        attack_dataset = AES_HD(root=data_dir, train=False)
        oracle_targets = ['label']
    elif dataset_name == 'otiait':
        from datasets.ed25519_wolfssl import ED25519
        profiling_dataset = ED25519(root=data_dir, train=True)
        attack_dataset = ED25519(root=data_dir, train=False)
        oracle_targets = ['label']
    elif dataset_name == 'otp':
        from datasets.one_truth_prevails import OneTruthPrevails
        profiling_dataset = OneTruthPrevails(root=data_dir, train=True)
        attack_dataset = OneTruthPrevails(root=data_dir, train=False)
        oracle_targets = ['label']
    else:
        assert False
    osnr = {}
    for target in oracle_targets:
        path = os.path.join(results_dir, 'first_order_parametric_statistical_assessment', f'attack_snr_{target}.npy')
        assessment = np.load(path)
        osnr[target] = assessment
    osnr = np.mean(np.stack(list(osnr.values())), axis=0)
    for idx, supervised_model_checkpoint in enumerate(supervised_model_checkpoints):
        lightning_module = SupervisedModule.load_from_checkpoint(supervised_model_checkpoint, map_location='cpu')
        classifier = lightning_module.classifier
        eval_on_attack_dataset(classifier, profiling_dataset, attack_dataset, dataset_name, os.path.join(output_dir, 'supervised', f'seed={idx}'))
        attribute_neural_net(
            classifier, profiling_dataset, attack_dataset, dataset_name, compute_gradvis=True, compute_occlusion=[1], output_dir=os.path.join(output_dir, 'supervised', f'seed={idx}')
        )
    print(f'Supervised classification performance on {dataset_name}:')
    evaluate_model_performance(os.path.join(output_dir, 'supervised'))
    print(f'Supervised localization performance on {dataset_name}:')
    evaluate_leakage_assessments(os.path.join(output_dir, 'supervised'), osnr)
    for idx, all_checkpoint in enumerate(all_checkpoints):
        lightning_module = ALLModule.load_from_checkpoint(all_checkpoint, map_location='cpu')
        classifier = WrapALLClassifier(lightning_module.cmi_estimator.classifiers)
        eval_on_attack_dataset(classifier, profiling_dataset, attack_dataset, dataset_name, os.path.join(output_dir, 'all', f'seed={idx}'))
        attribute_neural_net(
            classifier, profiling_dataset, attack_dataset, dataset_name, compute_gradvis=True, compute_occlusion=[1], output_dir=os.path.join(output_dir, 'all', f'seed={idx}')
        )
    print(f'ALL classification performance on {dataset_name}:')
    evaluate_model_performance(os.path.join(output_dir, 'all'))
    print(f'ALL localization performance on {dataset_name}')
    evaluate_leakage_assessments(os.path.join(output_dir, 'all'), osnr)
    print()
    print()