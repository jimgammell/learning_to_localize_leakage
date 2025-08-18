import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tqdm import tqdm
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
    hsweep_checkpoints = [
        os.path.join(results_dir, 'supervised_hparam_sweep', f'trial_{idx}', 'early_stop_checkpoint.ckpt') for idx in range(50)
    ]
    dropout_ablation_checkpoints = [
        os.path.join(results_dir, 'supervised_dropout_ablation_hparam_sweep', f'trial_{idx}', 'early_stop_checkpoint.ckpt') for idx in range(50)
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

    mttd_vals, loss_vals, rank_vals, agreement_vals = [], [], [], []
    for idx, ckpt in enumerate(dropout_ablation_checkpoints):
        lightning_module = SupervisedModule.load_from_checkpoint(ckpt, map_location='cpu')
        classifier = lightning_module.classifier
        eval_on_attack_dataset(classifier, profiling_dataset, attack_dataset, dataset_name, output_dir=os.path.dirname(ckpt), repeat_mttd_calculation=True, name='rep_attack_performance')
        if dataset_name not in ['otiait', 'otp']:
            mttd, loss, rank = evaluate_model_performance(os.path.dirname(ckpt), ret_vals=True, repeat_mttd_calculation=True)
            mttd_vals.append(float(mttd.mean()))
            loss_vals.append(float(loss))
            rank_vals.append(float(rank))
            leakage_assessment = np.load(os.path.join(os.path.dirname(ckpt), '1-occlusion.npz'))['attribution']
            agreement = spearmanr(leakage_assessment, osnr).statistic
            agreement_vals.append(agreement)
        else:
            loss, rank = evaluate_model_performance(os.path.dirname(ckpt), ret_vals=True)
            loss_vals.append(float(loss))
            rank_vals.append(float(rank))
            leakage_assessment = np.load(os.path.join(os.path.dirname(ckpt), '1-occlusion.npz'))['attribution']
            agreement = spearmanr(leakage_assessment, osnr).statistic
            agreement_vals.append(agreement)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].plot(loss_vals, agreement_vals, color='blue', marker='.', linestyle='none')
    axes[0].set_xlabel('Loss')
    axes[0].set_ylabel('Oracle Agreement (1-occlusion)')
    axes[0].set_xscale('log')
    axes[1].plot(rank_vals, agreement_vals, color='blue', marker='.', linestyle='none')
    axes[1].set_xlabel('Rank')
    axes[1].set_ylabel('Oracle Agreement (1-occlusion)')
    axes[1].set_xscale('log')
    if dataset_name not in ['otiait', 'otp']:
        axes[2].plot(mttd_vals, agreement_vals, color='blue', marker='.', linestyle='none')
        axes[2].set_xlabel('MTTD')
        axes[2].set_ylabel('Oracle Agreement (1-occlusion)')
        axes[2].set_xscale('log')
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'supervised_dropout_ablation_hparam_sweep', 'perf_vs_agreement.png'))
    plt.close(fig)
    if dataset_name not in ['otiait', 'otp']:
        boa_idx = np.nanargmax(agreement_vals)
        print(f'Best oracle agreement on {dataset_name}: MTTD={mttd_vals[boa_idx]},  Oracle agreement={agreement_vals[boa_idx]}')
        bmttd_idx = np.argmin(mttd_vals)
        print(f'Best MTTD on {dataset_name}: MTTD={mttd_vals[bmttd_idx]},  Oracle agreement={agreement_vals[bmttd_idx]}')
    else:
        boa_idx = np.nanargmax(agreement_vals)
        print(f'Best oracle agreement on {dataset_name}: rank={rank_vals[boa_idx]},  Oracle agreement={agreement_vals[boa_idx]}')
        brnk_idx = np.argmin(rank_vals)
        print(f'Best rank on {dataset_name}: rank={rank_vals[bmttd_idx]},  Oracle agreement={agreement_vals[bmttd_idx]}')

    mttd_vals, loss_vals, rank_vals, agreement_vals = [], [], [], []
    for idx, ckpt in enumerate(hsweep_checkpoints):
        lightning_module = SupervisedModule.load_from_checkpoint(ckpt, map_location='cpu')
        classifier = lightning_module.classifier
        eval_on_attack_dataset(classifier, profiling_dataset, attack_dataset, dataset_name, output_dir=os.path.dirname(ckpt))
        if dataset_name not in ['otiait', 'otp']:
            mttd, loss, rank = evaluate_model_performance(os.path.dirname(ckpt), ret_vals=True)
            mttd_vals.append(float(mttd))
            loss_vals.append(float(loss))
            rank_vals.append(float(rank))
            leakage_assessment = np.load(os.path.join(os.path.dirname(ckpt), '1-occlusion.npz'))['attribution']
            agreement = spearmanr(leakage_assessment, osnr).statistic
            agreement_vals.append(agreement)
        else:
            loss, rank = evaluate_model_performance(os.path.dirname(ckpt), ret_vals=True)
            loss_vals.append(float(loss))
            rank_vals.append(float(rank))
            leakage_assessment = np.load(os.path.join(os.path.dirname(ckpt), '1-occlusion.npz'))['attribution']
            agreement = spearmanr(leakage_assessment, osnr).statistic
            agreement_vals.append(agreement)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].plot(loss_vals, agreement_vals, color='blue', marker='.', linestyle='none')
    axes[0].set_xlabel('Loss')
    axes[0].set_ylabel('Oracle Agreement (1-occlusion)')
    axes[0].set_xscale('log')
    axes[1].plot(rank_vals, agreement_vals, color='blue', marker='.', linestyle='none')
    axes[1].set_xlabel('Rank')
    axes[1].set_ylabel('Oracle Agreement (1-occlusion)')
    axes[1].set_xscale('log')
    if dataset_name not in ['otiait', 'otp']:
        axes[2].plot(mttd_vals, agreement_vals, color='blue', marker='.', linestyle='none')
        axes[2].set_xlabel('MTTD')
        axes[2].set_ylabel('Oracle Agreement (1-occlusion)')
        axes[2].set_xscale('log')
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'supervised_hparam_sweep', 'perf_vs_agreement.png'))
    plt.close(fig)

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