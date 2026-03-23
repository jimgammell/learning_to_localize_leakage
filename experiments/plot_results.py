from pathlib import Path
from math import floor, log10
from collections import defaultdict
import pickle

import pandas
from scipy.stats import spearmanr
from matplotlib import pyplot as plt
from matplotlib.ticker import NullLocator

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
STEPS = {
    'ascadv1_fixed': 20000,
    'ascadv1_variable': 40000,
    'aes_hd': 20000,
    'dpav4': 10000,
    'otiait': 1000,
    'otp': 1000
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

def load_attack_curves(base_dir):
    attack_curves = defaultdict(lambda: defaultdict(list))
    collected_training_curves = defaultdict(list)
    for dataset_name in DATASET_NAMES.keys():
        for seed in [55, 56, 57, 58, 59]:
            if dataset_name in ['ascadv1_fixed', 'ascadv1_variable', 'dpav4', 'aes_hd']:
                attack_curve_path = os.path.join(
                    base_dir, dataset_name, 'supervised_models_for_attribution', 'classification', f'seed={seed}', 'attack_performance.npy'
                )
                if not os.path.exists(attack_curve_path):
                    print(f'Skipping file because it does not exist: {attack_curve_path}')
                    continue
                attack_curve = np.load(attack_curve_path)
                attack_curves[dataset_name]['ours'].append(attack_curve)
            with open(os.path.join(base_dir, dataset_name, 'supervised_models_for_attribution', 'classification', f'seed={seed}', 'training_curves.pickle'), 'rb') as f:
                training_curves = pickle.load(f)
            collected_training_curves[dataset_name].append({
                'train_loss': training_curves['train_loss'] if 'train_loss' in training_curves.keys() else training_curves['train_loss_step'],
                'val_loss': training_curves['val_loss'],
                'train_rank': training_curves['train_rank'] if 'train_rank' in training_curves.keys() else training_curves['train_rank_step'],
                'val_rank': training_curves['val_rank']
            })
        if os.path.exists(os.path.join(base_dir, dataset_name, 'pretrained_model_experiments')):
            model_dir = os.path.join(base_dir, dataset_name, 'pretrained_model_experiments')
            for subdir in os.listdir(model_dir):
                if os.path.exists(os.path.join(model_dir, subdir, 'attack_performance.npy')):
                    attack_performance = np.load(os.path.join(model_dir, subdir, 'attack_performance.npy'))
                    attack_curves[dataset_name][subdir].append(attack_performance)
                for seed_dir in [x for x in os.listdir(os.path.join(model_dir, subdir)) if x.split('=')[0] == 'seed']:
                    attack_performance = np.load(os.path.join(model_dir, subdir, seed_dir, 'attack_performance.npy'))
                    attack_curves[dataset_name][subdir].append(attack_performance)
    return attack_curves, collected_training_curves

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

def plot_attack_curves(base_dir, dest):
    attack_curvess, training_curvess = load_attack_curves(base_dir)
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot', (5, (10, 3))]
    markers = ['.', 'v', '^', 'P', '*']
    colors = ['blue', 'green', 'orange', 'red', 'brown']
    colors = {
        'ours': colors[0],
        'wouters': colors[1],
        'zaid': colors[2],
        'benadjila_cnn_best': colors[3],
        'benadjila': colors[3],
        'benadjila_mlp_best': colors[4]
    }
    to_method_label = {
        'ours': 'Ours',
        'wouters': 'WoutersNet',
        'zaid': 'ZaidNet',
        'benadjila': r"Benadjila's $\mathrm{CNN_{best}}$",
        'benadjila_cnn_best': r"Benadjila's $\mathrm{CNN_{best}}$",
        'benadjila_mlp_best': r"Benadjila's $\mathrm{MLP_{best}}$"
    }
    fontsize = 16
    fig, axes = plt.subplots(6, 3, figsize=(3*PLOT_WIDTH, 6*PLOT_WIDTH))
    for idx, dataset_name in enumerate(DATASET_NAMES.keys()):
        axes_r = axes[idx, :]
        training_curves = training_curvess[dataset_name]
        for sidx, training_curves_run in enumerate(training_curves):
            axes_r[0].plot(*training_curves_run['train_loss'], color=colors['ours'], linestyle=linestyles[sidx], alpha=0.25, label='train' if sidx==0 else None, **PLOT_KWARGS)
            axes_r[0].plot(*training_curves_run['val_loss'], color=colors['ours'], linestyle=linestyles[sidx], label='val' if sidx==0 else None, **PLOT_KWARGS)
            axes_r[1].plot(*training_curves_run['train_rank'], color=colors['ours'], linestyle=linestyles[sidx], alpha=0.25, label='train' if sidx==0 else None, **PLOT_KWARGS)
            axes_r[1].plot(*training_curves_run['val_rank'], color=colors['ours'], linestyle=linestyles[sidx], label='val' if sidx==0 else None, **PLOT_KWARGS)
        axes_r[0].set_xlabel('Training step', fontsize=fontsize)
        axes_r[0].set_ylabel('Cross-entropy loss $\downarrow$', fontsize=fontsize)
        axes_r[1].set_xlabel('Training step', fontsize=fontsize)
        axes_r[1].set_ylabel('Mean correct key rank $\downarrow$', fontsize=fontsize)
        if dataset_name in attack_curvess.keys():
            for method in attack_curvess[dataset_name]:
                if method == 'ours':
                    continue
                attack_curves = attack_curvess[dataset_name][method]
                for sidx, attack_curves_run in enumerate(attack_curves):
                    axes_r[2].plot(
                        np.arange(1, len(attack_curves_run)+1), attack_curves_run, color=colors[method], marker=markers[sidx], markersize=1, linestyle='-', linewidth=0.25,
                        label=to_method_label[method] if sidx==0 else None, **PLOT_KWARGS
                    )
                axes_r[2].set_xlabel('Traces seen', fontsize=fontsize)
                axes_r[2].set_ylabel('Correct AES key rank', fontsize=fontsize)
                axes_r[2].set_xscale('log')
            attack_curves = attack_curvess[dataset_name]['ours']
            for sidx, attack_curves_run in enumerate(attack_curves):
                axes_r[2].plot(
                    np.arange(1, len(attack_curves_run)+1), attack_curves_run, color=colors['ours'], marker=markers[sidx], markersize=1, linestyle='-', linewidth=0.25,
                    label=to_method_label['ours'] if sidx==0 else None, **PLOT_KWARGS
                )
            axes_r[2].set_xlabel('Traces seen', fontsize=fontsize)
            axes_r[2].set_ylabel('Correct AES key rank', fontsize=fontsize)
            axes_r[2].set_xscale('log')
        else:
            axes_r[2].axis('off')
        axes_r[0].legend(loc='upper right', fontsize=fontsize-2)
        axes_r[1].legend(loc='upper right', fontsize=fontsize-2)
        axes_r[0].set_title(f'Dataset: {DATASET_NAMES[dataset_name]}', fontsize=fontsize+2)
        axes_r[1].set_title(f'Dataset: {DATASET_NAMES[dataset_name]}', fontsize=fontsize+2)
        if dataset_name not in ['otiait', 'otp']:
            axes_r[2].legend(loc='upper right', fontsize=fontsize-2)
            axes_r[2].set_title(f'Dataset: {DATASET_NAMES[dataset_name]}', fontsize=fontsize+2)
    fig.tight_layout()
    fig.savefig(dest, **SAVEFIG_KWARGS)
    plt.close(fig)
    print(f'Saved training curve + rank vs. trace plots for the supervised models to {dest}.')

def plot_advll_training_curves(base_dir, dest):
    fontsize=16
    curvess = {}
    for dataset_name in DATASET_NAMES.keys():
        curves = defaultdict(list)
        for seed in [50, 51, 52, 53, 54]:
            model_dir = os.path.join(base_dir, dataset_name, 'all_runs', 'fair', f'seed={seed}')
            if dataset_name in ['ascadv1_fixed', 'ascadv1_variable', 'aes_hd']:
                with open(os.path.join(model_dir, 'classifier_pretraining', 'training_curves.pickle'), 'rb') as f:
                    pretrain_curves = pickle.load(f)
                with open(os.path.join(model_dir, 'all_training', 'training_curves.pickle'), 'rb') as f:
                    train_curves = pickle.load(f)
                for mname in ['train_theta_loss', 'val_theta_loss', 'train_theta_rank', 'val_theta_rank', 'oracle_snr_corr']:
                    curves[mname].append(np.concatenate([
                        pretrain_curves[mname][-1] if mname in pretrain_curves else pretrain_curves[f'{mname}_step'][-1] if mname != 'oracle_snr_corr' else np.zeros(len(train_curves['oracle_snr_corr'][1])),
                        train_curves[mname][-1] if mname in train_curves else train_curves[f'{mname}_step'][-1]
                    ]))
            else:
                with open(os.path.join(model_dir, 'all_training', 'training_curves.pickle'), 'rb') as f:
                    train_curves = pickle.load(f)
                for mname in ['train_theta_loss', 'val_theta_loss', 'train_theta_rank', 'val_theta_rank', 'oracle_snr_corr']:
                    curves[mname].append(train_curves[mname][-1] if mname in train_curves else train_curves[f'{mname}_step'][-1])
        curves = {k: np.stack(v) for k, v in curves.items()}
        curvess[dataset_name] = curves
    fig, axes = plt.subplots(6, 3, figsize=(3*PLOT_WIDTH, 6*PLOT_WIDTH))
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot', (5, (10, 3))]
    for dataset_name, axes_r in zip(DATASET_NAMES.keys(), axes):
        curves = curvess[dataset_name]
        if dataset_name in ['ascadv1_fixed', 'ascadv1_variable', 'aes_hd']:
            axes_r[0].axvspan(0, STEPS[dataset_name]/2, color='gray', alpha=0.1)
            axes_r[1].axvspan(0, STEPS[dataset_name]/2, color='gray', alpha=0.1)
            axes_r[2].axvspan(0, STEPS[dataset_name]/2, color='gray', alpha=0.1)
        for sidx in range(5):
            axes_r[0].plot(np.linspace(1, STEPS[dataset_name], len(curves['train_theta_loss'][sidx, :])), -curves['train_theta_loss'][sidx, :], color='blue', linestyle=linestyles[sidx], alpha=0.25, label='train' if sidx==0 else None)
            axes_r[0].plot(np.linspace(1, STEPS[dataset_name], len(curves['val_theta_loss'][sidx, :])), -curves['val_theta_loss'][sidx, :], color='blue', linestyle=linestyles[sidx], label='val' if sidx==0 else None)
            axes_r[1].plot(np.linspace(1, STEPS[dataset_name], len(curves['train_theta_rank'][sidx, :])), curves['train_theta_rank'][sidx, :], color='blue', linestyle=linestyles[sidx], alpha=0.25, label='train' if sidx==0 else None)
            axes_r[1].plot(np.linspace(1, STEPS[dataset_name], len(curves['val_theta_rank'][sidx, :])), curves['val_theta_rank'][sidx, :], color='blue', linestyle=linestyles[sidx], label='val' if sidx==0 else None)
            axes_r[2].plot(np.linspace(1, STEPS[dataset_name], len(curves['oracle_snr_corr'][sidx, :])), curves['oracle_snr_corr'][sidx, :], color='blue', linestyle=linestyles[sidx])
        axes_r[0].set_xlabel('Training step', fontsize=fontsize)
        axes_r[0].set_ylabel(r'MC estimate of $\mathcal{L}(\boldsymbol{\theta}, \tilde{\boldsymbol{\eta}})$ $\updownarrow$', fontsize=fontsize)
        axes_r[1].set_xlabel('Training step', fontsize=fontsize)
        axes_r[1].set_ylabel(r'Classifier mean rank $\downarrow$', fontsize=fontsize)
        axes_r[2].set_xlabel('Training step', fontsize=fontsize)
        axes_r[2].set_ylabel(r'Oracle agreement $\uparrow$', fontsize=fontsize)
        axes_r[0].legend(loc='upper right', fontsize=fontsize-2)
        axes_r[1].legend(loc='upper right', fontsize=fontsize-2)
        axes_r[0].set_title(f'Dataset: {DATASET_NAMES[dataset_name]}', fontsize=fontsize+2)
        axes_r[1].set_title(f'Dataset: {DATASET_NAMES[dataset_name]}', fontsize=fontsize+2)
        axes_r[2].set_title(f'Dataset: {DATASET_NAMES[dataset_name]}', fontsize=fontsize+2)
    fig.tight_layout()
    fig.savefig(dest)
    plt.close(fig)
    print(f'Saved training curves for ALL to {dest}.')

def plot_main_paper_qualitative_comparison(base_dir, dest):
    all_base_dir = os.path.join(base_dir, 'ascadv1_variable', 'all_runs', 'fair', 'seed=50', 'all_training')
    bl_base_dir = os.path.join(base_dir, 'ascadv1_variable', 'supervised_models_for_attribution', 'classification', 'seed=55')
    param_base_dir = os.path.join(base_dir, 'ascadv1_variable', 'first_order_parametric_statistical_assessment')
    ORACLE_PATHS = {
        r'$r_{\mathrm{in}}$': param_base_dir / r'attack_snr_r_in.npy',
        r'$r_{2}$': param_base_dir / r'attack_snr_r.npy',
        r'$r_{\mathrm{out}}$': param_base_dir / r'attack_snr_r_out.npy',
        r'$w_2 \oplus k_2 \oplus r_{\mathrm{in}}$': param_base_dir / r'attack_snr_plaintext__key__r_in.npy',
        r'$\operatorname{S}(w_2 \oplus k_2) \oplus r_2$': param_base_dir / r'attack_snr_subbytes__r.npy',
        r'$\operatorname{S}(w_2 \oplus k_2) \oplus r_{\mathrm{out}}$': param_base_dir / r'attack_snr_subbytes__r_out.npy',
        r'$S_{\mathrm{prev}} \oplus \operatorname{S}(w_2 \oplus k_2) \oplus r_{\mathrm{out}}$': param_base_dir / r'attack_snr_s_prev__subbytes__r_out.npy',
        r'Security load': param_base_dir / r'attack_snr_security_load.npy'
    }
    all_assessment = np.load(os.path.join(all_base_dir, 'leakage_assessment.npy'))
    baseline_assessment = np.load(os.path.join(bl_base_dir, '7-second-order-occlusion.npz'), allow_pickle=True)['attribution']
    oracle_assessments = {k: np.load(v) for k, v in oracle_paths.items()}

    def disable_xticks(ax):
        ax.tick_params(
            axis='x',
            which='both',
            labelbottom=False
        )
    def disable_yticks(ax):
        ax.tick_params(
            axis='y',
            which='both',
            labelleft=False
        )
    def disable_twinyticks(ax):
        ax.tick_params(
            axis='y',
            which='both',
            labelleft=False,
            labelright=False,
            left=False,
            right=True
        )

    fig = plt.figure(figsize=(16, 4))
    gs = fig.add_gridspec(2, 9, width_ratios=[1, 1, 0.2, 1, 1, 0.2, 1, 1, 1], height_ratios=[1, 1], wspace=0.25, hspace=0.25)
    vars_ax = fig.add_subplot(gs[0:2, 6:9])
    vars_tax = vars_ax.twinx()
    oracle_ax = fig.add_subplot(gs[0:2, 0:2])
    bl_ax = fig.add_subplot(gs[0, 3:5])
    all_ax = fig.add_subplot(gs[1, 3:5])
    composite_ax = fig.add_subplot(gs[0, 6])
    rin_ax = fig.add_subplot(gs[0, 7], sharex=composite_ax)
    rout_ax = fig.add_subplot(gs[1, 6], sharex=composite_ax)
    r_ax = fig.add_subplot(gs[1, 7], sharex=composite_ax)
    secl_ax = fig.add_subplot(gs[0, 8], sharex=composite_ax)
    aux_ax = fig.add_subplot(gs[1, 8], sharex=composite_ax)

    disable_xticks(composite_ax)
    disable_xticks(rin_ax)
    disable_xticks(secl_ax)
    disable_yticks(rin_ax)
    disable_yticks(secl_ax)
    disable_yticks(r_ax)
    disable_yticks(aux_ax)
    disable_xticks(bl_ax)

    def make_overlay_axis_inert(ax):
        ax.set_frame_on(False)
        ax.patch.set_visible(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(which='both', top=False, bottom=False, left=False, right=False,
                    labelbottom=False, labelleft=False, labelright=False)
        ax.xaxis.set_major_locator(NullLocator())
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
    make_overlay_axis_inert(vars_ax)
    make_overlay_axis_inert(vars_tax)

    vars_ax.set_xlabel(r'Leakiness from white-box assessment', labelpad=20)
    vars_ax.set_ylabel(r'Leakiness estimated by \textsf{ALL} (ours)', color='blue', labelpad=30)
    vars_tax.set_ylabel(r'Leakiness estimated by $2^{\mathrm{nd}}$-order 7-Occlusion', color='green', labelpad=25)

    oracle_ax.set_xlabel(r'Time $t$')
    oracle_ax.set_ylabel(r"`Oracle' leakiness of $X_t$")
    oracle_ax.set_title(r'White-box SNR assessment')
    kwargs = dict(marker='.', markersize=1, linestyle='-', linewidth=0.2, rasterized=True)
    oracle_kwargs = [
        {'color': 'red'},
        {'color': 'yellow'},
        {'color': 'teal'},
        {'color': 'green'},
        {'color': 'black'},
        {'color': 'blue'},
        {'color': 'teal', 'linestyle': 'dotted'},
        {'color': 'grey', 'linestyle': 'dotted'}
    ]
    for (target_var_name, oracle_snr), _oracle_kwargs in zip(oracle_assessments.items(), oracle_kwargs):
        _kwargs = copy(kwargs)
        _kwargs.update(_oracle_kwargs)
        oracle_ax.plot(oracle_snr, **_kwargs, label=target_var_name)
    oracle_ax.legend(loc='upper center', ncol=2, fontsize=8)
    oracle_ax.set_yscale('log')

    bl_ax.set_ylabel(r'Estimated leakiness of $X_t$')
    bl_ax.set_title(r'$2^{\mathrm{nd}}$-order 7-occlusion (best baseline)')
    bl_ax.plot(baseline_assessment, color='green', **kwargs)
    bl_ax.set_yscale('log')

    all_ax.set_xlabel(r'Time $t$')
    all_ax.set_ylabel(r'Estimated leakiness of $X_t$')
    all_ax.set_title(r'\textsf{ALL} (ours)')
    all_ax.plot(all_assessment, color='blue', **kwargs)

    all_kwargs = {'marker': 'o', 'markersize': 1, 'rasterized': True, 'linestyle': 'none'}
    bl_kwargs = {'marker': '^', 'markersize': 1, 'alpha': 0.5, 'rasterized': True, 'linestyle': 'none'}

    composite_assessment = np.stack(list(oracle_assessments.values())).mean(axis=0)
    composite_ax.set_title(r'All variables', fontsize=10)
    composite_ax.plot(composite_assessment, all_assessment, color='blue', **all_kwargs)
    composite_tax = composite_ax.twinx()
    composite_tax.plot(composite_assessment, baseline_assessment, color='green', **bl_kwargs)
    composite_ax.set_xscale('log')
    composite_tax.set_yscale('log')

    rin_assessment = np.stack([oracle_assessments[r'$r_{\mathrm{in}}$'], oracle_assessments[r'$w_2 \oplus k_2 \oplus r_{\mathrm{in}}$']]).mean(axis=0)
    rin_ax.set_title(r'$(r_{\mathrm{in}},\,w_2 \oplus k_2 \oplus r_{\mathrm{in}})$', fontsize=10)
    rin_tax = rin_ax.twinx()
    rin_ax.plot(rin_assessment, all_assessment, color='blue', **all_kwargs)
    rin_tax.plot(rin_assessment, baseline_assessment, color='green', **bl_kwargs)
    rin_ax.set_xscale('log')
    rin_tax.set_yscale('log')

    rout_assessment = np.stack([oracle_assessments[r'$r_{\mathrm{out}}$'], oracle_assessments[r'$\operatorname{S}(w_2 \oplus k_2) \oplus r_{\mathrm{out}}$']]).mean(axis=0)
    rout_ax.set_title(r'$(r_{\mathrm{out}},\,\operatorname{S}(w_2 \oplus k_2) \oplus r_{\mathrm{out}})$', fontsize=10)
    rout_tax = rout_ax.twinx()
    rout_ax.plot(rout_assessment, all_assessment, color='blue', **all_kwargs)
    rout_tax.plot(rout_assessment, baseline_assessment, color='green', **bl_kwargs)
    rout_ax.set_xscale('log')
    rout_tax.set_yscale('log')

    r_assessment = np.stack([oracle_assessments[r'$r_{2}$'], oracle_assessments[r'$\operatorname{S}(w_2 \oplus k_2) \oplus r_2$']]).mean(axis=0)
    r_ax.set_title(r'$(r_2,\,\operatorname{S}(w_2 \oplus k_2) \oplus r_2)$', fontsize=10)
    r_tax = r_ax.twinx()
    r_ax.plot(r_assessment, all_assessment, color='blue', **all_kwargs)
    r_tax.plot(r_assessment, baseline_assessment, color='green', **bl_kwargs)
    r_ax.set_xscale('log')
    r_tax.set_yscale('log')

    secl_assessment = oracle_assessments[r'Security load']
    secl_ax.set_title(r'Security load', fontsize=10)
    secl_tax = secl_ax.twinx()
    secl_ax.plot(secl_assessment, all_assessment, color='blue', **all_kwargs)
    secl_tax.plot(secl_assessment, baseline_assessment, color='green', **bl_kwargs)
    secl_ax.set_xscale('log')
    secl_tax.set_yscale('log')

    aux_assessment = oracle_assessments[r'$S_{\mathrm{prev}} \oplus \operatorname{S}(w_2 \oplus k_2) \oplus r_{\mathrm{out}}$']
    aux_ax.set_title(r'$S_{\mathrm{prev}} \oplus \operatorname{S}(w_2 \oplus k_2) \oplus r_{\mathrm{out}}$', fontsize=10)
    aux_tax = aux_ax.twinx()
    aux_ax.plot(aux_assessment, all_assessment, color='blue', **all_kwargs)
    aux_tax.plot(aux_assessment, baseline_assessment, color='green', **bl_kwargs)
    aux_ax.set_xscale('log')
    aux_tax.set_yscale('log')

    disable_twinyticks(composite_tax)
    disable_twinyticks(rin_tax)
    disable_twinyticks(rout_tax)
    disable_twinyticks(r_tax)

    fig.savefig(dest, dpi=150, bbox_inches='tight')
    plt.close(fig)

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

    # Generate Fig. 18 where we plot the training curves and rank vs. trace count for the supervised models.
    plot_attack_curves(OUTPUT_DIR, os.path.join(fig_dir, 'attack_curves.pdf'))

    # Generate Fig. 15 where we plot the training curves of ALL.
    plot_advll_training_curves(OUTPUT_DIR, os.path.join(fig_dir, 'all_training_curves.pdf'))

if __name__ == '__main__':
    main()