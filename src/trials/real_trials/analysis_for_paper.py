import os
from pathlib import Path
from collections import defaultdict
from math import floor, log10, ceil
from copy import copy
import pickle

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from matplotlib import pyplot as plt

from common import *

DATASET_NAMES = {
    'ascadv1_fixed': 'ASCADv1 (fixed)',
    'ascadv1_variable': 'ASCADv1 (variable)',
    'dpav4': 'DPAv4 (Zaid vsn.)',
    'aes_hd': 'AES-HD',
    'otiait': 'OTiAiT',
    'otp': 'OTP (1024-bit)'
}

METHOD_NAMES = {
    'random': 'Random',
    #'prof_oracle': 'Oracle (train set)',
    'gradvis': 'GradVis~\\cite{masure2019}',
    'saliency': 'Saliency~\\cite{simonyan2014, hettwer2020}',
    'inputxgrad': r'Input $*$ Grad~\cite{shrikumar2017, wouters2020}',
    'lrp': 'LRP~\\cite{bach2015, hettwer2020}',
    'occpoi': 'OccPOI~\\cite{yap2025}',
    '1-occlusion': '1-Occlusion~\\cite{zeiler2014, hettwer2020}',
    'm-occlusion': r'$m^*$-Occlusion~\cite{schamberger2023}',
    '1-second-order-occlusion': r'$1$-Occlusion$^2$~\cite{schamberger2023}',
    'm-second-order-occlusion': r'$m^*$-Occlusion$^2$~\cite{schamberger2023}',
    'all': 'ALL (ours)'
}

OPTIMAL_WINDOW_SIZES = {
    'ascadv1_fixed': 3,
    'ascadv1_variable': 7,
    'dpav4': 41,
    'aes_hd': 31,
    'otiait': 3,
    'otp': 5
}

OPTIMAL_ALL_HPARAMS = {
    'ascadv1_fixed': {'theta_lr': 7e-5,'etat_lr': 0.0049, 'gamma_bar': 0.9},
    'ascadv1_variable': {'theta_lr': 0.0004, 'etat_lr': 0.016, 'gamma_bar': 0.6},
    'dpav4': {'theta_lr': 9e-6, 'etat_lr': 0.00045, 'gamma_bar': 0.8},
    'aes_hd': {'theta_lr': 0.0001, 'gamma_bar': 0.85, 'etat_lr': 0.0003},
    'otiait': {'theta_lr': 5e-6, 'etat_lr': 0.0025, 'gamma_bar': 0.8},
    'otp': {'theta_lr': 7e-5, 'etat_lr': 0.0049, 'gamma_bar': 0.9}
}

def load_attack_curves(base_dir):
    dataset_names = ['ascadv1_fixed', 'ascadv1_variable', 'dpav4', 'aes_hd']
    attack_curves = {dataset_name: [] for dataset_name in dataset_names}
    for dataset_name in dataset_names:
        for seed in [55, 56, 57, 58, 59]:
            attack_curve_path = os.path.join(
                base_dir, dataset_name, 'supervised_models_for_attribution', 'classification', f'seed={seed}', 'attack_performance.npy'
            )
            if not os.path.exists(attack_curve_path):
                print(f'Skipping file because it does not exist: {attack_curve_path}')
                continue
            attack_curve = np.load(attack_curve_path)
            attack_curves[dataset_name].append(attack_curve)
    attack_curves = {k: np.stack(v) for k, v in attack_curves.items()}
    return attack_curves

def plot_attack_curves(base_dir, dest):
    attack_curves = load_attack_curves(base_dir)
    fig, axes = plt.subplots(1, len(attack_curves), figsize=(len(attack_curves)*PLOT_WIDTH, PLOT_WIDTH))
    for idx, (dataset_name, attack_curve) in enumerate(attack_curves.items()):
        ax = axes[idx]
        traces_seen = np.arange(1, attack_curve.shape[1]+1)
        ax.fill_between(traces_seen, np.min(attack_curve, axis=0), np.max(attack_curve, axis=0), color='blue', alpha=0.25, **PLOT_KWARGS)
        ax.plot(traces_seen, np.median(attack_curve, axis=0), color='blue', linestyle='-', **PLOT_KWARGS)
        ax.set_xlabel('Traces seen')
        ax.set_ylabel('Rank of correct key')
        ax.set_title(f'Dataset: {DATASET_NAMES[dataset_name]}')
        ax.set_xscale('log')
        ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(dest, **SAVEFIG_KWARGS)
    plt.close(fig)

def load_snr_assessments(base_dir):
    snr_assessments = {dataset_name: {'profiling': {}, 'attack': {}} for dataset_name in DATASET_NAMES.keys()}
    for dataset_name in DATASET_NAMES.keys():
        if dataset_name in ['ascadv1_fixed', 'ascadv1_variable']:
            targets = ['r_in', 'r', 'r_out', 'plaintext__key__r_in', 'subbytes__r', 'subbytes__r_out', 's_prev__subbytes__r_out', 'security_load']
        if dataset_name in ['dpav4', 'aes_hd', 'otiait', 'otp']:
            targets = ['label']
        snr_dir = os.path.join(base_dir, dataset_name, 'first_order_parametric_statistical_assessment')
        for phase in ['profiling', 'attack']:
            for target in targets:
                snr_path = os.path.join(snr_dir, f'{phase}_snr_{target}.npy')
                assert os.path.exists(snr_path), snr_path
                snr_assessments[dataset_name][phase][target] = np.load(snr_path)
    return snr_assessments

def get_oracle_assessments(base_dir, phase: Literal['profiling', 'attack'] = 'attack'):
    oracle_assessments = {}
    snr_assessments = load_snr_assessments(base_dir)
    for dataset_name in DATASET_NAMES.keys():
        oracle_assessments[dataset_name] = np.mean(np.stack(list(snr_assessments[dataset_name][phase].values())), axis=0)
    return oracle_assessments

def get_oracle_agreement(assessment, oracle):
    assert oracle.std() > 0
    if assessment.std() == 0:
        return 0.
    else:
        return spearmanr(assessment, oracle).statistic

def load_m_occlusion_oracle_agreement_scores(base_dir):
    dataset_names = list(DATASET_NAMES.keys())
    oracle_assessments = get_oracle_assessments(base_dir)
    occlusion_assessments = {dataset_name: defaultdict(list) for dataset_name in dataset_names}
    performance_vs_window_size = {dataset_name: defaultdict(list) for dataset_name in dataset_names}
    for dataset_name in dataset_names:
        oracle_assessment = oracle_assessments[dataset_name]
        for seed in [55, 56, 57, 58, 59]:
            window_sizes = np.arange(1, 21, 2) if dataset_name in ['ascadv1_fixed', 'ascadv1_variable', 'otiait', 'otp'] else np.arange(1, 51, 2)
            for window_size in window_sizes:
                assessment_path = os.path.join(base_dir, dataset_name, 'supervised_models_for_attribution', 'classification', f'seed={seed}', f'{window_size}-occlusion.npz')
                if not os.path.exists(assessment_path):
                    print(f'Skipping file because it does not exist: {assessment_path}')
                    continue
                assessment = np.load(assessment_path, allow_pickle=True)['attribution']
                oracle_agreement = get_oracle_agreement(assessment, oracle_assessment)
                occlusion_assessments[dataset_name][window_size].append(assessment)
                performance_vs_window_size[dataset_name][window_size].append(oracle_agreement)
    occlusion_assessments = {dataset_name: 
        {window_size: np.stack(occlusion_assessments[dataset_name][window_size]) for window_size in np.arange(1, 51, 2) if len(occlusion_assessments[dataset_name][window_size]) > 0}
    for dataset_name in dataset_names}
    performance_vs_window_size = {dataset_name: 
        {window_size: np.stack(performance_vs_window_size[dataset_name][window_size]) for window_size in np.arange(1, 51, 2) if len(performance_vs_window_size[dataset_name][window_size]) > 0}
    for dataset_name in dataset_names}
    max_key, max_val, vall = None, -float('inf'), None
    for key, val in performance_vs_window_size['aes_hd'].items():
        if val.mean() > max_val:
            max_val = val.mean()
            max_key = key
            vall = val
    return performance_vs_window_size, occlusion_assessments

def plot_m_occlusion_oracle_agreement_scores(base_dir, dest):
    oracle_agreement_scores, occlusion_assessments = load_m_occlusion_oracle_agreement_scores(base_dir)
    fig, axes = plt.subplots(6, 4, figsize=(4*PLOT_WIDTH, 6*PLOT_WIDTH))
    for idx, dataset_name in enumerate(oracle_agreement_scores.keys()):
        window_sizes = np.arange(1, 51, 2) if dataset_name in ['aes_hd', 'dpav4'] else np.arange(1, 21, 2)
        axes_r = axes[idx, :]
        agreement_scores = np.stack([oracle_agreement_scores[dataset_name][window_size] for window_size in window_sizes], axis=1)
        mean_score, std_score = agreement_scores.mean(axis=0), agreement_scores.std(axis=0)
        axes_r[0].fill_between(window_sizes, mean_score-std_score, mean_score+std_score, color='blue', alpha=0.25, **PLOT_KWARGS)
        axes_r[0].plot(window_sizes, mean_score, color='blue', **PLOT_KWARGS)
        axes_r[0].set_xlabel('Occlusion window size')
        axes_r[0].set_title('Oracle agreement')
        axes_r[0].set_ylabel(f'Dataset: {DATASET_NAMES[dataset_name]}')
        for idx, window_size in enumerate([1, 11, 19]):
            ax = axes_r[idx+1]
            assessment = occlusion_assessments[dataset_name][window_size]
            mean_assessment, std_assessment = np.mean(assessment, axis=0), np.std(assessment, axis=0)
            ax.fill_between(np.arange(len(mean_assessment)), mean_assessment-std_assessment, mean_assessment+std_assessment, color='blue', alpha=0.25, **PLOT_KWARGS)
            ax.plot(np.arange(len(mean_assessment)), mean_assessment, color='blue', **PLOT_KWARGS)
            ax.set_xlabel(r'Timestep $t$')
            ax.set_ylabel(r'Estimated leakiness of $X_t$')
            ax.set_title(f'Window size: {window_size}')
    fig.tight_layout()
    fig.savefig(dest, **SAVEFIG_KWARGS)
    plt.close(fig)

def plot_leakiness_assessments(base_dir, dest, only_ascadv1_variable: bool = False):
    dataset_names = list(DATASET_NAMES.keys()) if not only_ascadv1_variable else ['ascadv1_variable']
    cols = ceil(len(dataset_names)/2)
    rows = ceil(len(dataset_names)/cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*PLOT_WIDTH, rows*PLOT_WIDTH))
    if only_ascadv1_variable:
        axes = np.array([axes])
    oracle_assessments = get_oracle_assessments(base_dir)
    for idx, dataset_name in enumerate(dataset_names):
        ax = axes.flatten()[idx]
        tax = ax.twinx()
        oracle_assessment = oracle_assessments[dataset_name]
        all_path = os.path.join(base_dir, dataset_name, 'all_runs', 'fair', 'seed=50', 'all_training', 'leakage_assessment.npy')
        all_assessment = np.load(all_path)
        ax.plot(oracle_assessment, all_assessment, color='blue', linestyle='none', marker='s', markersize=5, alpha=0.5, label=r'\textbf{ALL (ours)}', **PLOT_KWARGS)
        occl_path = os.path.join(base_dir, dataset_name, 'supervised_models_for_attribution', 'classification', 'seed=55', f'{OPTIMAL_WINDOW_SIZES[dataset_name]}-occlusion.npz')
        occl = np.load(occl_path, allow_pickle=True)['attribution']
        tax.plot(oracle_assessment, occl, color='red', linestyle='none', marker='o', markersize=5, alpha=0.3, label=r'$m^*$-occlusion', **PLOT_KWARGS)
        occl2_path = os.path.join(base_dir, dataset_name, 'supervised_models_for_attribution', 'classification', 'seed=55', f'{OPTIMAL_WINDOW_SIZES[dataset_name]}-second-order-occlusion.npz')
        occl2 = np.load(occl2_path, allow_pickle=True)['attribution']
        occl2 += occl.min() - occl2.min()
        tax.plot(oracle_assessment, occl2, color='purple', linestyle='none', marker='+', markersize=6, alpha=0.5, label=r'$m^*$-occlusion$^2$', **PLOT_KWARGS)
        occpoi_path = os.path.join(base_dir, dataset_name, 'supervised_models_for_attribution', 'classification', 'seed=55', 'occpoi.npz')
        occpoi = np.load(occpoi_path, allow_pickle=True)['attribution']
        tax.plot(oracle_assessment, occpoi, color='orange', linestyle='none', marker='v', markersize=7, alpha=1, label='OccPOI', **PLOT_KWARGS)
        ax.set_xlabel('Oracle leakiness values', fontsize=16)
        ax.set_ylabel('ALL (ours) predictions', color='blue', fontsize=16)
        tax.set_ylabel('Baseline (w/ scale + shift) predictions', fontsize=16)
        ax.set_xscale('log')
        tax.set_yscale('log')
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = tax.get_legend_handles_labels()
        handles = h1 + h2
        labels = l1 + l2
        tax.legend(handles, labels, loc='lower right', fontsize=10, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(dest, **SAVEFIG_KWARGS)
    plt.close(fig)

def load_traces_over_time(base_dir):
    traces = defaultdict(lambda: defaultdict(list))
    for dataset_name in DATASET_NAMES.keys():
        for seed in [50, 51, 52, 53, 54]:
            all_curves_path = os.path.join(base_dir, dataset_name, 'all_runs', 'fair', f'seed={seed}', 'all_training', 'training_curves.pickle')
            if not os.path.exists(all_curves_path):
                print(f'Skipping file because it does not exist: {all_curves_path}')
                continue
            with open(all_curves_path, 'rb') as f:
                all_curves = pickle.load(f)
            oracle_agreement = all_curves['oracle_snr_corr'][-1]
            if dataset_name in ['ascadv1_fixed', 'ascadv1_variable', 'aes_hd']: # there is a pretraining phase
                oracle_agreement = np.concatenate([np.zeros(len(oracle_agreement)), oracle_agreement])
            traces[dataset_name]['all'].append(oracle_agreement)
        for seed in [55, 56, 57, 58, 59]:
            sup_curves_path = os.path.join(base_dir, dataset_name, 'attr_over_time', f'seed={seed}', 'training_curves.pickle')
            if not os.path.exists(sup_curves_path):
                print(f'Skipping file because it does not exist: {sup_curves_path}')
                continue
            with open(sup_curves_path, 'rb') as f:
                sup_curves = pickle.load(f)
            for method_name in ['gradvis', 'saliency', 'lrp', 'inputxgrad', '1-occlusion', 'm-occlusion']:
                oracle_agreement = sup_curves[f'{method_name}_oracle_agreement'][1]
                oracle_agreement = oracle_agreement[np.concatenate(([True], oracle_agreement[1:] != oracle_agreement[:-1]))]
                traces[dataset_name][method_name].append(oracle_agreement)
    traces = {dataset_name: {k: np.stack(v) for k, v in traces[dataset_name].items()} for dataset_name in DATASET_NAMES.keys()}
    return traces

def plot_traces_over_time(full_traces, dest, oracle_agreement_vals, only_ascadv1_variable: bool = False):
    to_kwargs = {
        'all': {'color': 'blue', 'label': r'\textbf{ALL (ours)}'},
        'gradvis': {'color': 'green', 'label': 'Gradient-based methods', 'markersize': 5, 'marker': 'o', 'alpha': 0.5},
        'saliency': {'color': 'green', 'markersize': 5, 'marker': 'o', 'alpha': 0.5},
        'lrp': {'color': 'green', 'markersize': 5, 'marker': 'o', 'alpha': 0.5},
        'inputxgrad': {'color': 'green', 'markersize': 5, 'marker': 'o', 'alpha': 0.5},
        '1-occlusion': {'color': 'red', 'label': '1-occlusion', 'markersize': 5, 'marker': 's', 'alpha': 0.8},
        'm-occlusion': {'color': 'red', 'label': r'$m^*$-occlusion', 'markersize': 5, 'marker': '.', 'alpha': 1.0},
        'occpoi': {'color': 'orange', 'label': 'OccPOI (final)'},
        'm-second-order-occlusion': {'color': 'purple', 'label': r'$m^*$-occlusion$^2$ (final)'}
    }
    dataset_names = list(DATASET_NAMES.keys()) if not only_ascadv1_variable else ['ascadv1_variable']
    cols = ceil(len(dataset_names)/2)
    rows = ceil(len(dataset_names)/cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*PLOT_WIDTH, rows*PLOT_WIDTH))
    if only_ascadv1_variable:
        axes = np.array([axes])
    for idx, dataset_name in enumerate(dataset_names):
        traces = full_traces[dataset_name]
        ax = axes.flatten()[idx]
        timesteps = 40000 if dataset_name == 'ascadv1_variable' else traces['all'].shape[1]
        for method_name, method_trace in traces.items():
            if method_name == 'all':
                ax.fill_between(
                    np.linspace(1, timesteps, method_trace.shape[1]), method_trace.mean(axis=0)-method_trace.std(axis=0), method_trace.mean(axis=0)+method_trace.std(axis=0), 
                    color='blue', alpha=0.25, **PLOT_KWARGS
                )
                ax.plot(np.linspace(1, timesteps, method_trace.shape[1]), method_trace.mean(axis=0), **to_kwargs['all'], **PLOT_KWARGS)
            else:
                ax.errorbar(np.linspace(1, timesteps, method_trace.shape[1]), method_trace.mean(axis=0), method_trace.std(axis=0), linewidth=2, elinewidth=2, capsize=5, **to_kwargs[method_name], **PLOT_KWARGS)
        ax.axhspan(
            ymin=oracle_agreement_vals[dataset_name]['occpoi'].mean()-oracle_agreement_vals[dataset_name]['occpoi'].std(),
            ymax=oracle_agreement_vals[dataset_name]['occpoi'].mean()+oracle_agreement_vals[dataset_name]['occpoi'].std(),
            alpha=0.25, xmin=0.0, xmax=1.0, **to_kwargs['occpoi'], **PLOT_KWARGS
        )
        ax.axhspan(
            ymin=oracle_agreement_vals[dataset_name]['m-second-order-occlusion'].mean()-oracle_agreement_vals[dataset_name]['m-second-order-occlusion'].std(),
            ymax=oracle_agreement_vals[dataset_name]['m-second-order-occlusion'].mean()+oracle_agreement_vals[dataset_name]['m-second-order-occlusion'].std(),
            alpha=0.25, xmin=0.0, xmax=1.0, **to_kwargs['m-second-order-occlusion'], **PLOT_KWARGS
        )
        ax.axhline(0.0, color='black', linestyle='--', label='Random', **PLOT_KWARGS)
        ax.set_xlabel('Training steps', fontsize=16)
        ax.set_ylabel(r'Oracle agreement $\uparrow$', fontsize=16)
        ax.legend(loc='upper left', ncol=2, fontsize='x-small', framealpha=0.5)
    fig.tight_layout()
    fig.savefig(dest, **SAVEFIG_KWARGS)
    plt.close(fig)

def load_all_sensitivity_analysis_data(base_dir):
    gamma_bar_sweep = defaultdict(lambda: defaultdict(list))
    theta_lr_scalar_sweep = defaultdict(lambda: defaultdict(list))
    etat_lr_scalar_sweep = defaultdict(lambda: defaultdict(list))
    oracle_assessments = get_oracle_assessments(base_dir)
    dataset_names = list(DATASET_NAMES.keys())
    for dataset_name in dataset_names:
        oracle_assessment = oracle_assessments[dataset_name]
        trial_dir = os.path.join(base_dir, dataset_name, 'all_sensitivity_analysis')
        for seed in [55, 56, 57, 58, 59]:
            if not os.path.exists(os.path.join(trial_dir, f'seed={seed}')):
                print(f'Skipping directory {os.path.join(trial_dir, f"seed={seed}")} because it does not exist')
                continue
            gamma_bar_subdirs = [x for x in os.listdir(os.path.join(trial_dir, f'seed={seed}')) if x.split('=')[0] == 'gamma_bar']
            for x in gamma_bar_subdirs:
                gamma_bar = float(x.split('=')[1])
                if os.path.exists(path := os.path.join(trial_dir, f'seed={seed}', x, 'all_training', 'leakage_assessment.npy')):
                    assessment = np.load(path)
                else:
                    print(f'Skipping path {path} because it does not exist')
                    continue
                oracle_agreement = get_oracle_agreement(assessment, oracle_assessment)
                gamma_bar_sweep[dataset_name][gamma_bar].append(oracle_agreement)
            theta_lr_scalar_subdirs = [x for x in os.listdir(os.path.join(trial_dir, f'seed={seed}')) if x.split('=')[0] == 'theta_lr_scalar']
            for x in theta_lr_scalar_subdirs:
                theta_lr_scalar = float(x.split('=')[1])
                if os.path.exists(path := os.path.join(trial_dir, f'seed={seed}', x, 'all_training', 'leakage_assessment.npy')):
                    assessment = np.load(path)
                else:
                    print(f'Skipping path {path} because it does not exist')
                    continue
                oracle_agreement = get_oracle_agreement(assessment, oracle_assessment)
                theta_lr_scalar_sweep[dataset_name][round(theta_lr_scalar, 3)].append(oracle_agreement)
            etat_lr_sweep_subdirs = [x for x in os.listdir(os.path.join(trial_dir, f'seed={seed}')) if x.split('=')[0] == 'etat_lr_scalar']
            for x in etat_lr_sweep_subdirs:
                etat_lr_scalar = float(x.split('=')[1])
                if os.path.exists(path := os.path.join(trial_dir, f'seed={seed}', x, 'all_training', 'leakage_assessment.npy')):
                    assessment = np.load(path)
                else:
                    print(f'Skipping path {path} because it does not exist')
                    continue
                oracle_agreement = get_oracle_agreement(assessment, oracle_assessment)
                etat_lr_scalar_sweep[dataset_name][round(etat_lr_scalar, 3)].append(oracle_agreement)
    for dataset_name in dataset_names:
        n = min(len(x) for x in gamma_bar_sweep[dataset_name].values())
        x = np.array(list(gamma_bar_sweep[dataset_name].keys()))
        y = np.stack([x[:n] for x in gamma_bar_sweep[dataset_name].values()])
        gamma_bar_sweep[dataset_name] = (x, y)
        n = min(len(x) for x in theta_lr_scalar_sweep[dataset_name].values())
        x = np.array(list(theta_lr_scalar_sweep[dataset_name].keys()))
        y = np.stack([x[:n] for x in theta_lr_scalar_sweep[dataset_name].values()])
        theta_lr_scalar_sweep[dataset_name] = (x, y)
        n = min(len(x) for x in etat_lr_scalar_sweep[dataset_name].values())
        x = np.array(list(etat_lr_scalar_sweep[dataset_name].keys()))
        y = np.stack([x[:n] for x in etat_lr_scalar_sweep[dataset_name].values()])
        etat_lr_scalar_sweep[dataset_name] = (x, y)
    return gamma_bar_sweep, theta_lr_scalar_sweep, etat_lr_scalar_sweep
    
def plot_all_sensitivity_analysis(gamma_bar_sweep, theta_lr_scalar_sweep, etat_lr_scalar_sweep, dest, only_ascadv1_variable: bool = False):
    dataset_names = ['ascadv1_variable'] if only_ascadv1_variable else list(DATASET_NAMES.keys())
    cols = ceil(len(dataset_names)/2)
    rows = ceil(len(dataset_names)/cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*PLOT_WIDTH, rows*PLOT_WIDTH))
    if only_ascadv1_variable:
        axes = np.array([axes])
    for dataset_name, ax in zip(dataset_names, axes.flatten()):
        if not only_ascadv1_variable:
            ax.set_title(f'Dataset: {DATASET_NAMES[dataset_name]}')
        theta_lr_scalar_ax = ax.twiny()
        etat_lr_scalar_ax = ax.twiny()
        etat_lr_scalar_ax.spines['top'].set_position(('outward', 40))
        ax.axhline(0., color='black', linestyle='--')

        gamma_bar_vals, gamma_bar_agreements = gamma_bar_sweep[dataset_name]
        sorted_indices = np.argsort(gamma_bar_vals)
        gamma_bar_vals = gamma_bar_vals[sorted_indices]
        gamma_bar_agreements = gamma_bar_agreements[sorted_indices]
        ax.fill_between(gamma_bar_vals, gamma_bar_agreements.mean(axis=1)-gamma_bar_agreements.std(axis=1), gamma_bar_agreements.mean(axis=1)+gamma_bar_agreements.std(axis=1), color='blue', alpha=0.25, **PLOT_KWARGS)
        ax.plot(gamma_bar_vals, gamma_bar_agreements.mean(axis=1), color='blue', marker='.', linestyle='none', **PLOT_KWARGS)
        ax.set_xlabel(r'Budget hyperparameter $\overline{\gamma}$', color='blue', fontsize=16)
        ax.set_ylabel(r'Oracle agreement $\uparrow$', fontsize=16)

        theta_lr_scalar_vals, theta_lr_scalar_agreements = theta_lr_scalar_sweep[dataset_name]
        sorted_indices = np.argsort(theta_lr_scalar_vals)
        theta_lr_vals = theta_lr_scalar_vals[sorted_indices]*OPTIMAL_ALL_HPARAMS[dataset_name]['theta_lr']
        theta_lr_scalar_agreements = theta_lr_scalar_agreements[sorted_indices]
        theta_lr_scalar_ax.fill_between(theta_lr_vals, theta_lr_scalar_agreements.mean(axis=1)-theta_lr_scalar_agreements.std(axis=1), theta_lr_scalar_agreements.mean(axis=1)+theta_lr_scalar_agreements.std(axis=1), color='red', alpha=0.25, **PLOT_KWARGS)
        theta_lr_scalar_ax.plot(theta_lr_vals, theta_lr_scalar_agreements.mean(axis=1), color='red', marker='.', linestyle='none', **PLOT_KWARGS)

        etat_lr_scalar_vals, etat_lr_scalar_agreements = etat_lr_scalar_sweep[dataset_name]
        sorted_indices = np.argsort(etat_lr_scalar_vals)
        etat_lr_vals = etat_lr_scalar_vals[sorted_indices]*OPTIMAL_ALL_HPARAMS[dataset_name]['etat_lr']
        etat_lr_scalar_agreements = etat_lr_scalar_agreements[sorted_indices]
        etat_lr_scalar_ax.fill_between(etat_lr_vals, etat_lr_scalar_agreements.mean(axis=1)-etat_lr_scalar_agreements.std(axis=1), etat_lr_scalar_agreements.mean(axis=1)+etat_lr_scalar_agreements.std(axis=1), color='green', alpha=0.25, **PLOT_KWARGS)
        etat_lr_scalar_ax.plot(etat_lr_vals, etat_lr_scalar_agreements.mean(axis=1), color='green', marker='.', linestyle='none', **PLOT_KWARGS)

        theta_lr_scalar_ax.set_xlabel(r'Learning rate of $\boldsymbol{\theta}$', color='red', fontsize=16)
        etat_lr_scalar_ax.set_xlabel(r'Learning rate of $\tilde{\boldsymbol{\eta}}$', color='green', fontsize=16)
        theta_lr_scalar_ax.set_xscale('log')
        etat_lr_scalar_ax.set_xscale('log')
    fig.tight_layout()
    fig.savefig(dest, **SAVEFIG_KWARGS)
    plt.close(fig)

def get_assessments(base_dir):
    dataset_names = list(DATASET_NAMES.keys())
    assessments = {dataset_name: defaultdict(list) for dataset_name in dataset_names}
    assessment_runtimes = {dataset_name: defaultdict(list) for dataset_name in dataset_names}
    for dataset_name in dataset_names:
        for seed in [55, 56, 57, 58, 59]:
            sup_dir = os.path.join(base_dir, dataset_name, 'supervised_models_for_attribution', 'classification', f'seed={seed}')
            for method_name in [
                'gradvis', 'inputxgrad', 'lrp', 'occpoi', 'saliency', '1-second-order-occlusion', f'{OPTIMAL_WINDOW_SIZES[dataset_name]}-second-order-occlusion',
                '1-occlusion', f'{OPTIMAL_WINDOW_SIZES[dataset_name]}-occlusion'
            ]:
                assessment_path = os.path.join(sup_dir, f'{method_name}.npz')
                if not os.path.exists(assessment_path):
                    print(f'Skipping file because it does not exist: {assessment_path}')
                    continue
                rv = np.load(assessment_path, allow_pickle=True)
                assessment = rv['attribution']
                elapsed_time_ms = rv['elapsed_time']
                elapsed_time_min = 1e-3*elapsed_time_ms/60
                assessments[dataset_name][method_name].append(assessment)
                assessment_runtimes[dataset_name][method_name].append(elapsed_time_min)
        for seed in [50, 51, 52, 53, 54]:
            all_dir = os.path.join(base_dir, dataset_name, 'all_runs', 'fair', f'seed={seed}')
            assessment_path = os.path.join(all_dir, 'all_training', 'leakage_assessment.npy')
            time_path = os.path.join(all_dir, 'training_time.npy')
            if not os.path.exists(assessment_path):
                print(f'Skipping file because it does not exist: {assessment_path}')
                continue
            if not os.path.exists(time_path):
                print(f'Skipping file because it does not exist: {time_path}')
                continue
            assessment = np.load(assessment_path)
            elapsed_time_ms = np.load(time_path)
            elapsed_time_min = 1e-3*elapsed_time_ms/60
            assessments[dataset_name]['all'].append(assessment)
            assessment_runtimes[dataset_name]['all'].append(elapsed_time_min)
    assessments = {dataset_name: {key: np.stack(val) for key, val in assessments[dataset_name].items()} for dataset_name in dataset_names}
    assessment_runtimes = {dataset_name: {key: np.stack(val) for key, val in assessment_runtimes[dataset_name].items()} for dataset_name in dataset_names}
    return assessments, assessment_runtimes

def get_oracle_agreement_vals(base_dir):
    assessments, _ = get_assessments(base_dir)
    oracle_assessments = get_oracle_assessments(base_dir)
    profiling_oracle_assessments = get_oracle_assessments(base_dir, phase='profiling')
    data = {dataset_name: {method_name: None for method_name in METHOD_NAMES.keys()} for dataset_name in DATASET_NAMES.keys()}
    for dataset_name in DATASET_NAMES.keys():
        print(f'Dataset: {dataset_name}')
        oracle_assessment = oracle_assessments[dataset_name]
        profiling_oracle_assessment = profiling_oracle_assessments[dataset_name]
        data[dataset_name]['random'] = np.stack([get_oracle_agreement(np.random.rand(oracle_assessment.shape[-1]), oracle_assessment) for _ in range(5)])
        #data[dataset_name]['prof_oracle'] = get_oracle_agreement(oracle_assessment, profiling_oracle_assessment).reshape(1, -1)
        per_method_assessments = assessments[dataset_name]
        for assessment_name, assessment in per_method_assessments.items():
            agreement_vals = np.array([get_oracle_agreement(_assessment, oracle_assessment) for _assessment in assessment])
            if assessment_name in data[dataset_name].keys():
                data[dataset_name][assessment_name] = agreement_vals
            if assessment_name.split('-')[-1] == 'occlusion' and 'second-order' not in assessment_name:
                if data[dataset_name]['m-occlusion'] is None or agreement_vals.mean() >= data[dataset_name]['m-occlusion'].mean():
                    data[dataset_name]['m-occlusion'] = agreement_vals
            if assessment_name == f'{OPTIMAL_WINDOW_SIZES[dataset_name]}-second-order-occlusion':
                data[dataset_name]['m-second-order-occlusion'] = agreement_vals
            print(f'\t{assessment_name}: {agreement_vals.mean()} +/- {agreement_vals.std()}')
    return data

def get_dnn_occlusion_curves(base_dir):
    fwd_base_data = {dataset_name: defaultdict(list) for dataset_name in DATASET_NAMES.keys()}
    fwd_data = {dataset_name: {method_name: None for method_name in METHOD_NAMES.keys()} for dataset_name in DATASET_NAMES.keys()}
    rev_base_data = {dataset_name: defaultdict(list) for dataset_name in DATASET_NAMES.keys()}
    rev_data = {dataset_name: {method_name: None for method_name in METHOD_NAMES.keys()} for dataset_name in DATASET_NAMES.keys()}
    for dataset_name in DATASET_NAMES.keys():
        print(f'Dataset: {dataset_name}')
        for seed in [55, 56, 57, 58, 59]:
            subdir = os.path.join(base_dir, dataset_name, 'supervised_models_for_attribution', 'classification', f'seed={seed}')
            method_names = [x.split('_dnno')[0] for x in os.listdir(subdir) if x.split('_')[-1] == 'dnno.npz']
            for method_name in method_names:
                rv = np.load(os.path.join(subdir, f'{method_name}_dnno.npz'), allow_pickle=True)
                fwd_dnno = rv['fwd_dnno']
                rev_dnno = rv['rev_dnno']
                print(f'\tFWD AUC for {method_name}: {fwd_dnno.mean()} +/- {fwd_dnno.std()}')
                print(f'\tREV AUC for {method_name}: {rev_dnno.mean()} +/- {rev_dnno.std()}')
                fwd_base_data[dataset_name][method_name].append(rv['fwd_dnno'])
                rev_base_data[dataset_name][method_name].append(rv['rev_dnno'])
    for dataset_name in DATASET_NAMES:
        for k, v in fwd_base_data[dataset_name].items():
            print(dataset_name, k, [x.shape for x in v])
    fwd_base_data = {dataset_name: {k: np.stack(v) for k, v in fwd_base_data[dataset_name].items()} for dataset_name in DATASET_NAMES.keys()}
    rev_base_data = {dataset_name: {k: np.stack(v) for k, v in rev_base_data[dataset_name].items()} for dataset_name in DATASET_NAMES.keys()}
    for dataset_name in DATASET_NAMES.keys():
        for method_name in fwd_base_data[dataset_name].keys():
            if method_name in METHOD_NAMES.keys():
                fwd_data[dataset_name][method_name] = fwd_base_data[dataset_name][method_name]
                rev_data[dataset_name][method_name] = rev_base_data[dataset_name][method_name]
            fwd_data[dataset_name]['m-occlusion'] = fwd_base_data[dataset_name][f'{OPTIMAL_WINDOW_SIZES[dataset_name]}-occlusion']
            rev_data[dataset_name]['m-occlusion'] = rev_base_data[dataset_name][f'{OPTIMAL_WINDOW_SIZES[dataset_name]}-occlusion']
    return fwd_data, rev_data

# done with the help of ChatGPT
def create_performance_comparison_table(base_dir, dest, data):
    def to_one_sigfig(x):
        if x == 0 or np.isnan(x):
            return 0.
        exp = int(floor(log10(abs(x))))
        return round(x, -exp)
    def fmt(mean, std, should_highlight=False, should_underline=False):
        if mean is None or std is None:
            return r'n/a'
        error = to_one_sigfig(std)
        if error > 0:
            decimals = abs(int(floor(log10(error)))) if error < 1 else 0.
            val = round(mean, decimals)
            fmt_str = f'{{:.{decimals}f}} \\pm {{:.{decimals}f}}'
            rv = fmt_str.format(val, error)
        else:
            decimals = 3
            val = round(mean, 3)
            fmt_str = f'{{:.{decimals}f}}'
            rv = fmt_str.format(val)
        rv = '$' + rv + '$'
        if should_highlight:
            rv = f'\\best{{{rv}}}'
        if should_underline:
            rv = f'\\underline{{{rv}}}'
        return rv
    def build_full_tabular(latex_body, n_rows):
        header = (
            "\\begin{tabular}{lcccccc}\n"
            "\\toprule\n"
            "& \\multicolumn{2}{c}{\\textbf{2nd-order datasets}} "
            "\\multicolumn{4}{c}{\\textbf{1st-order datasets}} \\\\\n"
            "\\textbf{Method} & ASCADv1 (fixed)~\\cite{benadjila2020} "
            "& ASCADv1 (random)~\\cite{benadjila2020} "
            "& DPAv4 (Zaid vsn.)~\\cite{bhasin2014, zaid2020} "
            "& AES-HD~\\cite{bhasin2020} "
            "& OTiAiT~\\cite{weissbart2019} "
            "& OTP (1024-bit)~\\cite{saito2022} \\\\\n"
            "\\cmidrule{1-1} \\cmidrule(lr){2-3} \\cmidrule(lr){4-7}\n"
        )
        lines = latex_body.splitlines()
        start = next(i for i, ln in enumerate(lines) if ln.strip() == r"\midrule") + 1
        end   = next(i for i, ln in enumerate(lines[::-1]) if ln.strip() == r"\bottomrule")
        body_lines = lines[start: len(lines) - end - 1]
        for idx, body_line in enumerate(body_lines):
            if ('Random' in body_line) or (r'$m^*$-Occlusion$^2$' in body_line):
                body_lines[idx] += '\\midrule'
        body_lines = [f"{ln.lstrip()}" for ln in body_lines]
        footer = "\\bottomrule\n\\end{tabular}\n"
        return header + "\n".join(body_lines) + footer
    table = pd.DataFrame(index=list(METHOD_NAMES.values()), columns=list(DATASET_NAMES.values()))
    for dataset_name, subdata in data.items():
        best_method_idx = np.argmax([x.mean() if (x is not None and name != 'prof_oracle') else -np.inf for name, x in subdata.items()])
        best_name = list(subdata.keys())[best_method_idx]
        best_data = subdata[best_name]
        methods_to_underline = [
            method_name for method_name, method_data in subdata.items()
            if (method_data is not None)
            and (method_name != best_name)
            and (method_data.mean()+method_data.std() >= best_data.mean()-best_data.std())
            and method_name != 'prof_oracle'
        ]
        for method_name, method_data in subdata.items():
            should_highlight = method_name == best_name
            should_underline = method_name in methods_to_underline
            table.at[METHOD_NAMES[method_name], DATASET_NAMES[dataset_name]] = (
                fmt(method_data.mean(), method_data.std(), should_highlight=should_highlight, should_underline=should_underline) if method_data is not None
                else fmt(None, None, should_highlight=should_highlight)
            )
    latex_body = table.to_latex(
        escape=False,
        column_format='lcccccc',
        index_names=False,
        header=False
    )
    full_table = build_full_tabular(latex_body, n_rows=len(table))
    Path(dest).write_text(full_table)

def create_toy_gaussian_plot(base_dir, dest):
    def get_negative_rank(assessment):
        return np.sum(assessment[1:] <= assessment[0]) / len(assessment)
    def plot_trace(d, ax, **kwargs):
        leaky_points = np.array(list(d.keys()))
        negative_ranks = np.stack(list(d.values()))
        #_kwargs = copy(kwargs)
        #_kwargs['marker'] = 'none'
        #ax.plot(2**leaky_points, np.median(negative_ranks, axis=1), linestyle='-', linewidth=0.25, **_kwargs)
        ax.errorbar(
            2**leaky_points, np.median(negative_ranks, axis=1), yerr=(np.median(negative_ranks, axis=1)-negative_ranks.min(axis=1), negative_ranks.max(axis=1)-np.median(negative_ranks, axis=1)),
            linewidth=0.25, elinewidth=0.25, capsize=2, **kwargs
        )
        #for _negative_ranks in negative_ranks.transpose():
        #    ax.plot(2**leaky_points, _negative_ranks, linestyle='none', alpha=0.25, **kwargs)
    negative_ranks = defaultdict(lambda: defaultdict(list))
    for seed in [0, 1, 2, 3, 4]:
        for leaky_point_count in range(14):
            data_dir = os.path.join(base_dir, 'toy_gaussian', 'first_order', f'seed={seed}', f'leaky_pair_count={leaky_point_count}')
            if not os.path.exists(path := os.path.join(data_dir, 'parametric', 'param_assessments.npz')):
                print(f'Skipping path because it does not exist: {path}')
            param_assessments = np.load(path, allow_pickle=True)
            for key, val in param_assessments.items():
                negative_ranks[key][leaky_point_count].append(get_negative_rank(val))
            if not os.path.exists(path := os.path.join(data_dir, 'advll', 'leakage_assessment.npy')):
                print(f'Skipping path because it does not exist: {path}')
            all_assessment = np.load(path)
            negative_ranks['all'][leaky_point_count].append(get_negative_rank(all_assessment))
            if not os.path.exists(path := os.path.join(data_dir, 'supervised', 'early_stop_leakage_assessments.npz')):
                print(f'Skipping path because it does not exist: {path}')
            sup_assessments = np.load(path, allow_pickle=True)
            for key, val in sup_assessments.items():
                negative_ranks[key][leaky_point_count].append(get_negative_rank(val))
    negative_ranks = {method: {k: np.stack(v) for k, v in negative_ranks[method].items()} for method in negative_ranks.keys()}
    fig, ax = plt.subplots(1, 1, figsize=(PLOT_WIDTH, PLOT_WIDTH))
    groups = {
        'parametric': ['snr', 'sosd', 'cpa'],
        'gradient': ['gradvis', 'saliency', 'lrp', 'inputxgrad'],
        'occlusion': [f'{m}-occlusion' for m in np.arange(1, 21, 2)],
        'occpoi': ['occpoi'],
        'all': ['all']
    }
    to_kwargs = {
        'parametric': {'color': 'green', 'label': 'Best parametric', 'markersize': 5, 'marker': 'o'},
        'gradient': {'color': 'purple', 'label': 'Best gradient-based', 'markersize': 3, 'marker': 's'},
        'occlusion': {'color': 'cyan', 'label': r'Best $m$-occlusion', 'markersize': 5, 'marker': 'X'},
        'occpoi': {'color': 'orange', 'label': 'OccPOI', 'markersize': 3, 'marker': 'v'},
        'all': {'color': 'blue', 'label': r'\textbf{ALL (ours)}', 'markersize': 5, 'marker': '^'}
    }
    def get_trace_for_group(group):
        best_trace = {leaky_point_count: np.full((5,), np.inf, dtype=np.float32) for leaky_point_count in range(14)}
        for leaky_point_count in range(14):
            for method in group:
                if not leaky_point_count in negative_ranks[method]:
                    continue
                vals = negative_ranks[method][leaky_point_count]
                if vals.mean() < best_trace[leaky_point_count].mean():
                    best_trace[leaky_point_count] = vals
            if not np.all(np.isfinite(best_trace[leaky_point_count])):
                del best_trace[leaky_point_count]
        return best_trace
    for group_name, group in groups.items():
        trace = get_trace_for_group(group)
        plot_trace(trace, ax, **to_kwargs[group_name], **PLOT_KWARGS)
    ax.axhline(0.0, color='black', linestyle='--', label='Oracle')
    ax.axhline(0.5, color='red', linestyle='--', label='Random')
    ax.set_xlabel(r'Number of second-order leaky pairs: $D$')
    ax.set_ylabel(r'False negative rate $\downarrow$')
    ax.set_xscale('log')
    ax.legend(ncol=2, loc='upper left', framealpha=0.5, fontsize='x-small')
    fig.tight_layout()
    fig.savefig(dest, **SAVEFIG_KWARGS)

def do_analysis_for_paper():
    fig_dir = os.path.join(OUTPUT_DIR, 'plots_for_paper')
    os.makedirs(fig_dir, exist_ok=True)
    print('Plotting occlusion window size sweeps...')
    plot_m_occlusion_oracle_agreement_scores(OUTPUT_DIR, os.path.join(fig_dir, 'occl_window_size_sweep.pdf'))
    print()
    plot_leakiness_assessments(OUTPUT_DIR, os.path.join(fig_dir, 'qualitative_relationship_with_oracle.pdf'))
    plot_leakiness_assessments(OUTPUT_DIR, os.path.join(fig_dir, 'ascadv1_variable_relationship_with_oracle.pdf'), only_ascadv1_variable=True)
    oracle_agreement_vals = get_oracle_agreement_vals(OUTPUT_DIR)
    print('Creating oracle agreement table...')
    create_performance_comparison_table(OUTPUT_DIR, os.path.join(fig_dir, 'oracle_agreement_table'), oracle_agreement_vals)
    print()
    create_toy_gaussian_plot(OUTPUT_DIR, os.path.join(fig_dir, 'toy_gaussian_plot.pdf'))
    gamma_bar_sweep, theta_lr_scalar_sweep, etat_lr_scalar_sweep = load_all_sensitivity_analysis_data(OUTPUT_DIR)
    plot_all_sensitivity_analysis(gamma_bar_sweep, theta_lr_scalar_sweep, etat_lr_scalar_sweep, os.path.join(fig_dir, 'ascadv1_variable_sensitivity_analysis.pdf'), only_ascadv1_variable=True)
    plot_all_sensitivity_analysis(gamma_bar_sweep, theta_lr_scalar_sweep, etat_lr_scalar_sweep, os.path.join(fig_dir, 'all_sensitivity_analysis.pdf'))
    traces = load_traces_over_time(OUTPUT_DIR)
    plot_traces_over_time(traces, os.path.join(fig_dir, 'traces_over_time.pdf'), oracle_agreement_vals, only_ascadv1_variable=True)
    print('Plotting attack curves...')
    plot_attack_curves(OUTPUT_DIR, os.path.join(fig_dir, 'attack_curves.pdf'))
    print()
    r"""print('Creating DNN occlusion AUC table...')
    fwd_dnno_data, rev_dnno_data = get_dnn_occlusion_curves(OUTPUT_DIR)
    create_performance_comparison_table(OUTPUT_DIR, os.path.join(fig_dir, 'fwd_dnno_auc_table'), fwd_dnno_data)
    create_performance_comparison_table(OUTPUT_DIR, os.path.join(fig_dir, 'rev_dnno_auc_table'), rev_dnno_data)
    print()"""