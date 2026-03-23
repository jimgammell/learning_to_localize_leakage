from pathlib import Path
from math import floor, log10
from collections import defaultdict

import pandas
from scipy.stats import spearmanr
from matplotlib import pyplot as plt

from leakage_localization.common import *
from leakage_localization.trials.synthetic_data_experiments import Trial as SyntheticTrial
from init_things import *

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
    'snr': 'SNR',
    'sosd': 'SOSD',
    'cpa': 'CPA',
    'gradvis': 'GradVis',
    'saliency': 'Saliency',
    'inputxgrad': r'Input $*$ Grad',
    'lrp': 'LRP',
    'occpoi': 'OccPOI',
    '1-occlusion': '1-Occlusion',
    'm-occlusion': r'$m^*$-Occlusion',
    '1-second-order-occlusion': r'$1$-Occlusion$^2$',
    'm-second-order-occlusion': r'$m^*$-Occlusion$^2$'
}
for method_name in ['1-occlusion', 'm-occlusion', 'gradvis', 'inputxgrad', 'saliency']:
    METHOD_NAMES[f'wouters-{method_name}'] = 'WoutersNet ' + METHOD_NAMES[method_name]
    METHOD_NAMES[f'zaid-{method_name}'] = 'ZaidNet ' + METHOD_NAMES[method_name]
METHOD_NAMES['all'] = r'\textsf{ALL} (ours)'

OPTIMAL_WINDOW_SIZES = {
    'ascadv1_fixed': 3,
    'ascadv1_variable': 7,
    'dpav4': 41,
    'aes_hd': 31,
    'otiait': 3,
    'otp': 5
}

def load_eval_metrics(base_dir):
    oracle_agreement = defaultdict(lambda: defaultdict(list))
    fwd_dnno_auc = defaultdict(lambda: defaultdict(list))
    rev_dnno_auc = defaultdict(lambda: defaultdict(list))
    ta_mttd = defaultdict(lambda: defaultdict(list))
    for dataset_name in DATASET_NAMES.keys():
        param_dir = os.path.join(base_dir, dataset_name, 'first_order_parametric_statistical_assessment')
        for seed in range(5):
            random_metrics = np.load(os.path.join(param_dir, f'random_evaluation_metrics_{seed}.npz'), allow_pickle=True)
            oracle_agreement[dataset_name]['random'].append(random_metrics['oracle_agreement'])
            fwd_dnno_auc[dataset_name]['random'].append(random_metrics['fwd_dnno'])
            rev_dnno_auc[dataset_name]['random'].append(random_metrics['rev_dnno'])
            ta_mttd[dataset_name]['random'].append(random_metrics['ta_ttd'] + (1 if dataset_name in ['otiait', 'otp'] else 0))
        for method_name in ['snr', 'sosd', 'cpa']:
            eval_metrics = np.load(os.path.join(param_dir, f'{method_name}_evaluation_metrics.npz'), allow_pickle=True)
            oracle_agreement[dataset_name][method_name].append(eval_metrics['oracle_agreement'])
            fwd_dnno_auc[dataset_name][method_name].append(eval_metrics['fwd_dnno'])
            rev_dnno_auc[dataset_name][method_name].append(eval_metrics['rev_dnno'])
            ta_mttd[dataset_name][method_name].append(eval_metrics['ta_ttd'] + (1 if dataset_name in ['otiait', 'otp'] else 0))
        pretrained_method_dir = os.path.join(base_dir, dataset_name, 'pretrained_model_experiments')
        pretrained_method_names = ['1-occlusion', f'{OPTIMAL_WINDOW_SIZES[dataset_name]}-occlusion', 'gradvis', 'inputxgrad', 'saliency']
        if os.path.exists(pretrained_method_dir):
            for subdir in os.listdir(pretrained_method_dir):
                paths = [os.path.join(pretrained_method_dir, subdir, f'seed={x}') for x in range(5)]
                for _path in paths:
                    for method_name in pretrained_method_names:
                        path = os.path.join(_path, f'{method_name}_evaluation_metrics.npz')
                        if not os.path.exists(path):
                            continue
                        eval_metrics = np.load(path, allow_pickle=True)
                        if method_name == f'{OPTIMAL_WINDOW_SIZES[dataset_name]}-occlusion':
                            method_name = 'm-occlusion'
                        if method_name == f'{OPTIMAL_WINDOW_SIZES[dataset_name]}-second-order-occlusion':
                            method_name = 'm-second-order-occlusion'
                        method_name = subdir + '-' + method_name
                        oracle_agreement[dataset_name][method_name].append(eval_metrics['oracle_agreement'])
                        fwd_dnno_auc[dataset_name][method_name].append(eval_metrics['fwd_dnno'])
                        rev_dnno_auc[dataset_name][method_name].append(eval_metrics['rev_dnno'])
                        ta_mttd[dataset_name][method_name].append(eval_metrics['ta_ttd'] + (1 if dataset_name in ['otiait', 'otp'] else 0))
        for seed in range(55, 60):
            attr_dir = os.path.join(base_dir, dataset_name, 'supervised_models_for_attribution', 'classification', f'seed={seed}')
            for method_name in [
                'gradvis', 'inputxgrad', 'lrp', 'saliency', 'occpoi',
                '1-occlusion', f'{OPTIMAL_WINDOW_SIZES[dataset_name]}-occlusion', '1-second-order-occlusion', f'{OPTIMAL_WINDOW_SIZES[dataset_name]}-second-order-occlusion'
            ]:
                path = os.path.join(attr_dir, f'{method_name}_evaluation_metrics.npz')
                if not os.path.exists(path):
                    continue
                eval_metrics = np.load(path, allow_pickle=True)
                if method_name == f'{OPTIMAL_WINDOW_SIZES[dataset_name]}-occlusion':
                    method_name = 'm-occlusion'
                if method_name == f'{OPTIMAL_WINDOW_SIZES[dataset_name]}-second-order-occlusion':
                    method_name = 'm-second-order-occlusion'
                oracle_agreement[dataset_name][method_name].append(eval_metrics['oracle_agreement'])
                fwd_dnno_auc[dataset_name][method_name].append(eval_metrics['fwd_dnno'])
                rev_dnno_auc[dataset_name][method_name].append(eval_metrics['rev_dnno'])
                ta_mttd[dataset_name][method_name].append(eval_metrics['ta_ttd'] + (1 if dataset_name in ['otiait', 'otp'] else 0))
        for seed in range(50, 55):
            path = os.path.join(base_dir, dataset_name, 'all_runs', 'fair', f'seed={seed}', 'evaluation_metrics.npz')
            if not os.path.exists(path):
                print(f'DNE: {path}')
                continue
            eval_metrics = np.load(path, allow_pickle=True)
            oracle_agreement[dataset_name]['all'].append(eval_metrics['oracle_agreement'])
            fwd_dnno_auc[dataset_name]['all'].append(eval_metrics['fwd_dnno'])
            rev_dnno_auc[dataset_name]['all'].append(eval_metrics['rev_dnno'])
            ta_mttd[dataset_name]['all'].append(eval_metrics['ta_ttd'] + (1 if dataset_name in ['otiait', 'otp'] else 0))
    oracle_agreement = {k1: {k2: np.stack(v) for k2, v in oracle_agreement[k1].items()} for k1 in oracle_agreement}
    fwd_dnno_auc = {k1: {k2: np.stack(v).mean(axis=1) for k2, v in fwd_dnno_auc[k1].items()} for k1 in fwd_dnno_auc}
    rev_dnno_auc = {k1: {k2: np.stack(v).mean(axis=1) for k2, v in rev_dnno_auc[k1].items()} for k1 in rev_dnno_auc}
    ta_mttd = {k1: {k2: np.stack(v) for k2, v in ta_mttd[k1].items()} for k1 in ta_mttd}
    print('Loaded performance evaluations for each method/dataset pair:')
    for dataset_name in DATASET_NAMES.keys():
        print(f'\tDataset: {dataset_name}')
        print('\t\tOracle agreement values:')
        for k, v in oracle_agreement[dataset_name].items():
            print(f'\t\t\t{k}: {v.mean()} +/- {v.std()}')
        print('\t\tFwd DNNO AUC values:')
        for k, v in fwd_dnno_auc[dataset_name].items():
            print(f'\t\t\t{k}: {v.mean()} +/- {v.std()}')
        print('\t\tRev DNNO AUC values:')
        for k, v in rev_dnno_auc[dataset_name].items():
            print(f'\t\t\t{k}: {v.mean()} +/- {v.std()}')
        print('\t\tTA MTD values:')
        for k, v in ta_mttd[dataset_name].items():
            print(f'\t\t\t{k}: {v.mean()} +/- {v.std()}')
    return oracle_agreement, fwd_dnno_auc, rev_dnno_auc, ta_mttd

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

# i.e. the average SNR assessment over sensitive variables
def load_oracle_assessments(base_dir, phase: Literal['profiling', 'attack'] = 'attack'):
    oracle_assessments = {}
    snr_assessments = load_snr_assessments(base_dir)
    for dataset_name in DATASET_NAMES.keys():
        oracle_assessments[dataset_name] = np.mean(np.stack(list(snr_assessments[dataset_name][phase].values())), axis=0)
    return oracle_assessments

def load_all_assessments(base_dir, selection_method='oracle'):
    dataset_names = list(DATASET_NAMES.keys())
    assessments = defaultdict(list)
    for dataset_name in dataset_names:
        for seed in [50, 51, 52, 53, 54]:
            all_dir = os.path.join(base_dir, dataset_name, 'all_runs', selection_method, f'seed={seed}')
            assessment_path = os.path.join(all_dir, 'all_training', 'leakage_assessment.npy')
            time_path = os.path.join(all_dir, 'training_time.npy')
            if not os.path.exists(assessment_path):
                print(f'Skipping file because it does not exist: {assessment_path}')
                continue
            if not os.path.exists(time_path):
                print(f'Skipping file because it does not exist: {time_path}')
                continue
            assessment = np.load(assessment_path)
            assessments[dataset_name].append(assessment)
    assessments = {k: np.stack(v) for k, v in assessments.items()}
    return assessments

def calculate_oracle_agreement(assessment, oracle):
    assert oracle.std() > 0
    if assessment.std() == 0:
        return 0.
    else:
        return spearmanr(assessment, oracle).statistic

def create_performance_comparison_table(dest, data, bigger_is_better: bool = True):
    def to_one_sigfig(x):
        if x == 0 or np.isnan(x):
            return 0.
        exp = int(floor(log10(abs(x))))
        return round(x, -exp)
    def fmt(mean, std, should_highlight=False, should_underline=False):
        if mean is None or std is None or np.isnan(mean) or np.isnan(std):
            return r'n/a'
        error = to_one_sigfig(std)
        if error > 0:
            decimals = abs(int(floor(log10(error)))) if error < 1 else 0
            val = round(mean, decimals)
            fmt_str = f'{{:.{decimals}f}} \\pm {{:.{decimals}f}}'
            rv = fmt_str.format(val, error)
        else:
            decimals = 3
            val = round(mean, 3)
            fmt_str = f'{{:.{decimals}f}}'
            rv = fmt_str.format(val)
        rv = '$' + rv + '$'
        if should_underline:
            rv = f'\\underline{{{rv}}}'
        if should_highlight:
            rv = f'\\best{{{rv}}}'
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
            if ('Random' in body_line) or (r'$m^*$-Occlusion$^2$' in body_line) or ('CPA' in body_line) or ('WoutersNet Saliency' in body_line) or ('ZaidNet Saliency' in body_line):
                body_lines[idx] += '\\midrule'
        body_lines = [f"{ln.lstrip()}" for ln in body_lines]
        footer = "\\bottomrule\n\\end{tabular}\n"
        return header + "\n".join(body_lines) + footer
    table = pandas.DataFrame(index=list(METHOD_NAMES.values()), columns=list(DATASET_NAMES.values()))
    for method_name in METHOD_NAMES:
        for dataset_name in DATASET_NAMES:
            table.at[METHOD_NAMES[method_name], DATASET_NAMES[dataset_name]] = r'n/a'
    for dataset_name, subdata in data.items():
        if bigger_is_better:
            vals = [x if (x is not None and name != 'prof_oracle') else -np.inf for name, x in subdata.items()]
            best_data = vals[np.argmax([x.mean() if (x is not None and name != 'prof_oracle') else -np.inf for name, x in subdata.items()])]
            vals = [x if (x is not None and name not in ['prof_oracle', 'snr', 'sosd', 'cpa', 'random']) else -np.inf for name, x in subdata.items()]
            best_dl_data = vals[np.argmax([x.mean() if (x is not None and name not in ['prof_oracle', 'snr', 'sosd', 'cpa', 'random']) else -np.inf for name, x in subdata.items()])]
        else:
            vals = [x if (x is not None and name != 'prof_oracle') else -np.inf for name, x in subdata.items()]
            best_data = vals[np.argmin([x.mean() if (x is not None and name != 'prof_oracle') else np.inf for name, x in subdata.items()])]
            vals = [x if (x is not None and name not in ['prof_oracle', 'snr', 'sosd', 'cpa', 'random']) else -np.inf for name, x in subdata.items()]
            best_dl_data = vals[np.argmin([x.mean() if (x is not None and name not in ['prof_oracle', 'snr', 'sosd', 'cpa', 'random']) else np.inf for name, x in subdata.items()])]
        methods_to_box = [
            method_name for method_name, method_data in subdata.items()
            if (method_data is not None)
            and (method_name not in ['prof_oracle'])
            and ((method_data.mean() >= best_data.mean()-best_data.std()) if bigger_is_better else (method_data.mean() <= best_data.mean()+best_data.std()))
        ]
        methods_to_underline = [
            method_name for method_name, method_data in subdata.items()
            if (method_data is not None)
            and (method_name not in ['snr', 'sosd', 'cpa', 'random', 'prof_oracle'])
            and ((method_data.mean() >= best_dl_data.mean()-best_dl_data.std()) if bigger_is_better else (method_data.mean() <= best_dl_data.mean()+best_dl_data.std()))
        ]
        for method_name, method_data in subdata.items():
            should_highlight = method_name in methods_to_box
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
    print(f'Saved performance comparison table to {dest}.')

def plot_leakiness_assessment_comparison_with_oracle(base_dir, dest):
    fontsize = 16
    assessmentss = load_all_assessments(base_dir)
    snr_assessmentss = load_snr_assessments(base_dir)
    oracle_assessments = load_oracle_assessments(base_dir)
    fig, axes = plt.subplots(6, 3, figsize=(3*PLOT_WIDTH, 6*PLOT_WIDTH))
    for idx, dataset_name in enumerate(DATASET_NAMES.keys()):
        if dataset_name in ['ascadv1_fixed', 'ascadv1_variable']:
            var_to_kwargs = {
                'plaintext__key__r_in': {'label': r'$w_2 \oplus k_2 \oplus r_{\mathrm{in}}$', 'color': 'green', 'linestyle': '-'},
                'r_in': {'label': r'$r_{\mathrm{in}}$', 'color': 'red', 'linestyle': '-'},
                'r_out': {'label': r'$r_{\mathrm{out}}$', 'color': 'teal', 'linestyle': '-'},
                'r': {'label': r'$r_2$', 'color': 'yellow', 'linestyle': '-'},
                'subbytes__r_out': {'label': r'$\operatorname{Sbox}(w_2 \oplus k_2) \oplus r_{\mathrm{out}}$', 'color': 'blue', 'linestyle': '-'},
                'subbytes__r': {'label': r'$\operatorname{Sbox}(w_2 \oplus k_2) \oplus r_2$', 'color': 'black', 'linestyle': '-'},
                's_prev__subbytes__r_out': {'label': r'$S_{\mathrm{prev}} \oplus \operatorname{Sbox}(w_2 \oplus k_2) \oplus r_{\mathrm{out}}$', 'color': 'teal', 'linestyle': '--'},
                'security_load': {'label': r'$\text{Security load}$', 'color': 'gray', 'linestyle': '--'}
            }
        elif dataset_name == 'dpav4':
            var_to_kwargs = {'label': {'label': r'$\operatorname{Sbox}(k_0 \oplus w_0) \oplus m_0$', 'color': 'blue', 'linestyle': '-'}}
        elif dataset_name == 'aes_hd':
            var_to_kwargs = {'label': {'label': r'$\operatorname{Sbox}^{-1}(k_{11}^* \oplus c_{11}) \oplus c_7$', 'color': 'blue', 'linestyle': '-'}}
        elif dataset_name == 'otiait':
            var_to_kwargs = {'label': {'label': 'Ephemeral key nibble', 'color': 'blue', 'linestyle': '-'}}
        elif dataset_name == 'otp':
            var_to_kwargs = {'label': {'label': 'Dummy load?', 'color': 'blue', 'linestyle': '-'}}
        else:
            assert False
        assessments = assessmentss[dataset_name]
        snr_assessments = snr_assessmentss[dataset_name]['attack']
        oracle_assessment = oracle_assessments[dataset_name]
        axes_r = axes[idx, :]
        for k, v in snr_assessments.items():
            axes_r[0].plot(v, **var_to_kwargs[k], linewidth=0.25, marker='.', markersize=2, **PLOT_KWARGS)
        axes_r[1].plot(assessments[0, :], color='blue', linestyle='-', linewidth=0.25, marker='.', markersize=2, **PLOT_KWARGS)
        axes_r[2].plot(oracle_assessment, assessments[0, :], color='blue', linestyle='none', marker='.', markersize=2, **PLOT_KWARGS)
        axes_r[0].legend(fontsize=8, ncol=2, loc='upper center')
        axes_r[0].set_xlabel(r'Time $t$', fontsize=fontsize)
        axes_r[0].set_ylabel(r'Oracle leakiness of $X_t$', fontsize=fontsize)
        axes_r[0].set_title(f'Dataset: {DATASET_NAMES[dataset_name]}', fontsize=fontsize+2)
        axes_r[0].set_yscale('log')
        axes_r[1].set_xlabel(r'Time $t$', fontsize=fontsize)
        axes_r[1].set_ylabel(r'Estimated leakiness of $X_t$ by ALL', fontsize=fontsize)
        axes_r[1].set_title(f'Dataset: {DATASET_NAMES[dataset_name]}', fontsize=fontsize+2)
        axes_r[2].set_xlabel(r'Oracle leakiness of $X_t$', fontsize=fontsize)
        axes_r[2].set_ylabel(r'Estimated leakiness of $X_t$ by ALL', fontsize=fontsize)
        axes_r[2].set_title(f'Dataset: {DATASET_NAMES[dataset_name]}', fontsize=fontsize+2)
        axes_r[2].set_xscale('log')
    fig.tight_layout()
    fig.savefig(dest, **SAVEFIG_KWARGS)
    plt.close(fig)
    print(f'Saved qualitative ALL vs. oracle leakiness comparison to {dest}.')

def plot_synthetic_dataset_experiments(base_dir, dest):
    synthetic_trial = SyntheticTrial(logging_dir=os.path.join(base_dir, 'synthetic'))
    synthetic_trial.plot_main_paper_sweeps(dest, dont_subsample=True)
    print(f'Saved synthetic experiment plots to {dest}.')

def main():
    fig_dir = os.path.join(OUTPUT_DIR, 'plots_for_paper')
    os.makedirs(fig_dir, exist_ok=True)

    # Generate Latex code for Tables 11--14 where we tabulate the performance of the method/dataset pairs on various datasets.
    #   Table 2 in the main paper is a condensed version of these.
    oracle_agreement, fwd_dnn_auc, rev_dnn_auc, ta_mtd = load_eval_metrics(OUTPUT_DIR)
    create_performance_comparison_table(os.path.join(fig_dir, 'oracle_agreement_table'), oracle_agreement)
    create_performance_comparison_table(os.path.join(fig_dir, 'fwd_dnno_table'), fwd_dnn_auc)
    create_performance_comparison_table(os.path.join(fig_dir, 'rev_dnno_table'), rev_dnn_auc)
    create_performance_comparison_table(os.path.join(fig_dir, 'ta_mtd_table'), ta_mtd)

    # Generate Fig. 3 where we plot the performance on the toy Gaussian dataset for various methods.

    # Generate Fig. 11 where we plot the ALL-based leakage assessments for synthetic datasets.
    #   Fig. 4 in the main paper is a condensed version of this.
    plot_synthetic_dataset_experiments(OUTPUT_DIR, os.path.join(fig_dir, 'synthetic_experiments.pdf'))

    # Generate Fig. 14 where we plot the oracle leakiness vs. the estimated leakiness by ALL.
    plot_leakiness_assessment_comparison_with_oracle(OUTPUT_DIR, os.path.join(fig_dir, 'qualitative_comparison_all_vs_oracle.pdf'))


if __name__ == '__main__':
    main()