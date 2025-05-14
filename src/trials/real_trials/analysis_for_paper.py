import os
from pathlib import Path
from collections import defaultdict
from math import floor, log10

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
    'prof_oracle': 'Oracle (train set)',
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
    'dpav4': 19,
    'aes_hd': 19,
    'otiait': 3,
    'otp': 5
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
            for window_size in np.arange(1, 21, 2):
                assessment_path = os.path.join(base_dir, dataset_name, 'supervised_models_for_attribution', 'classification', f'seed={seed}', f'{window_size}-occlusion.npz')
                if not os.path.exists(assessment_path):
                    print(f'Skipping file because it does not exist: {assessment_path}')
                    continue
                assessment = np.load(assessment_path, allow_pickle=True)['attribution']
                oracle_agreement = get_oracle_agreement(assessment, oracle_assessment)
                occlusion_assessments[dataset_name][window_size].append(assessment)
                performance_vs_window_size[dataset_name][window_size].append(oracle_agreement)
    occlusion_assessments = {dataset_name: {window_size: np.stack(occlusion_assessments[dataset_name][window_size]) for window_size in np.arange(1, 21, 2)} for dataset_name in dataset_names}
    performance_vs_window_size = {dataset_name: {window_size: np.stack(performance_vs_window_size[dataset_name][window_size]) for window_size in np.arange(1, 21, 2)} for dataset_name in dataset_names}
    return performance_vs_window_size, occlusion_assessments

def plot_m_occlusion_oracle_agreement_scores(base_dir, dest):
    oracle_agreement_scores, occlusion_assessments = load_m_occlusion_oracle_agreement_scores(base_dir)
    window_sizes = np.arange(1, 21, 2)
    fig, axes = plt.subplots(6, 4, figsize=(4*PLOT_WIDTH, 6*PLOT_WIDTH))
    for idx, dataset_name in enumerate(oracle_agreement_scores.keys()):
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
            ax.fill_between(np.arange(len(mean_assessment)), mean_assessment-std_assessment, mean_assessment+std_assessment, color='blue', alpha=0.25)
            ax.plot(np.arange(len(mean_assessment)), mean_assessment, color='blue')
            ax.set_xlabel(r'Timestep $t$')
            ax.set_ylabel(r'Estimated leakiness of $X_t$')
            ax.set_title(f'Window size: {window_size}')
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
            for method_name in ['gradvis', 'inputxgrad', 'lrp', 'occpoi', 'saliency', '1-second-order-occlusion', *[f'{m}-occlusion' for m in np.arange(1, 21, 2)]]:
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
        data[dataset_name]['prof_oracle'] = get_oracle_agreement(oracle_assessment, profiling_oracle_assessment).reshape(1, -1)
        per_method_assessments = assessments[dataset_name]
        for assessment_name, assessment in per_method_assessments.items():
            agreement_vals = np.array([get_oracle_agreement(_assessment, oracle_assessment) for _assessment in assessment])
            if assessment_name in data[dataset_name].keys():
                data[dataset_name][assessment_name] = agreement_vals
            if assessment_name.split('-')[-1] == 'occlusion':
                if data[dataset_name]['m-occlusion'] is None or agreement_vals.mean() >= data[dataset_name]['m-occlusion'].mean():
                    data[dataset_name]['m-occlusion'] = agreement_vals
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
    def fmt(mean, std, should_highlight=False):
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
        return rv
    def build_full_tabular(latex_body, n_rows):
        header = (
            "\\begin{tabular}{clcccccc}\n"
            "\\toprule\n"
            "& & \\multicolumn{2}{c}{\\textbf{2nd-order datasets}} "
            "& \\multicolumn{4}{c}{\\textbf{1st-order datasets}} \\\\\n"
            "& \\textbf{Method} & ASCADv1 (fixed)~\\cite{benadjila2020} "
            "& ASCADv1 (random)~\\cite{benadjila2020} "
            "& DPAv4 (Zaid vsn.)~\\cite{bhasin2014, zaid2020} "
            "& AES-HD~\\cite{bhasin2020} "
            "& OTiAiT~\\cite{weissbart2019} "
            "& OTP (1024-bit)~\\cite{saito2022} \\\\\n"
            "\\cmidrule{2-2} \\cmidrule(lr){3-4} \\cmidrule(lr){5-8}\n"
            f"\\multirow{{{n_rows}}}{{2cm}}{{Spearman's rank correlation with oracle leakiness}}\n"
        )
        lines = latex_body.splitlines()
        start = next(i for i, ln in enumerate(lines) if ln.strip() == r"\midrule") + 1
        end   = next(i for i, ln in enumerate(lines[::-1]) if ln.strip() == r"\bottomrule")
        body_lines = lines[start: len(lines) - end - 1]
        for idx, body_line in enumerate(body_lines):
            if ('Oracle' in body_line) or (r'$m^*$-$2^{\mathrm{nd}}$-order Occlusion' in body_line):
                body_lines[idx] += '\\cmidrule{2-8}'
        body_lines = [f"& {ln.lstrip()}" for ln in body_lines]
        footer = "\\bottomrule\n\\end{tabular}\n"
        return header + "\n".join(body_lines) + footer
    table = pd.DataFrame(index=list(METHOD_NAMES.values()), columns=list(DATASET_NAMES.values()))
    for dataset_name, subdata in data.items():
        best_method_idx = np.argmax([x.mean() if (x is not None and name != 'prof_oracle') else -np.inf for name, x in subdata.items()])
        best_name = list(subdata.keys())[best_method_idx]
        best_data = subdata[best_name]
        methods_to_highlight = [
            method_name for method_name, method_data in subdata.items()
            if (method_data is not None)
            and (method_data.mean()+method_data.std() >= best_data.mean()-best_data.std())
            and method_name != 'prof_oracle'
        ]
        for method_name, method_data in subdata.items():
            should_highlight = method_name in methods_to_highlight
            table.at[METHOD_NAMES[method_name], DATASET_NAMES[dataset_name]] = (
                fmt(method_data.mean(), method_data.std(), should_highlight=should_highlight) if method_data is not None
                else fmt(None, None, should_highlight=should_highlight)
            )
    latex_body = table.to_latex(
        escape=False,
        column_format='clcccccc',
        index_names=False,
        header=False
    )
    full_table = build_full_tabular(latex_body, n_rows=len(table))
    Path(dest).write_text(full_table)

def do_analysis_for_paper():
    fig_dir = os.path.join(OUTPUT_DIR, 'plots_for_paper')
    os.makedirs(fig_dir, exist_ok=True)
    print('Plotting attack curves...')
    plot_attack_curves(OUTPUT_DIR, os.path.join(fig_dir, 'attack_curves.pdf'))
    print()
    print('Plotting occlusion window size sweeps...')
    plot_m_occlusion_oracle_agreement_scores(OUTPUT_DIR, os.path.join(fig_dir, 'occl_window_size_sweep.pdf'))
    print()
    print('Creating oracle agreement table...')
    create_performance_comparison_table(OUTPUT_DIR, os.path.join(fig_dir, 'oracle_agreement_table'), get_oracle_agreement_vals(OUTPUT_DIR))
    print()
    print('Creating DNN occlusion AUC table...')
    fwd_dnno_data, rev_dnno_data = get_dnn_occlusion_curves(OUTPUT_DIR)
    create_performance_comparison_table(OUTPUT_DIR, os.path.join(fig_dir, 'fwd_dnno_auc_table'), fwd_dnno_data)
    create_performance_comparison_table(OUTPUT_DIR, os.path.join(fig_dir, 'rev_dnno_auc_table'), rev_dnno_data)
    print()