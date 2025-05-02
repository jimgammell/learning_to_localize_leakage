import tempfile
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import imageio

from common import *
from trials.utils import *

def plot_leakage_assessment(oracle_assessment: np.ndarray, leakage_assessment: np.ndarray, dest: str):
    timestep_count = leakage_assessment.shape[-1]
    leakage_assessment = leakage_assessment.reshape(-1, timestep_count)
    seed_count = leakage_assessment.shape[0]
    oracle_assessment.shape == (timestep_count,)
    fig, axes = plt.subplots(1, 2, figsize=(2*PLOT_WIDTH, PLOT_WIDTH))
    axes[0].set_xlabel('Timestep $t$')
    axes[0].set_ylabel('Estimated `leakiness\' of $X_t$')
    if seed_count > 1:
        mean, std = leakage_assessment.mean(axis=0), leakage_assessment.std(axis=0)
        axes[0].fill_between(np.arange(timestep_count), mean-std, mean+std, color='blue', alpha=0.25)
    axes[0].plot(np.arange(timestep_count), leakage_assessment.mean(axis=0), linestyle='none', marker='.', markersize=1, color='blue')
    axes[1].set_xlabel('Oracle SNR of $X_t$')
    axes[1].set_ylabel('Estimated `leakiness\' of $X_t$')
    sorted_indices = oracle_assessment.argsort()
    if seed_count > 1:
        axes[1].fill_between(oracle_assessment[sorted_indices], (mean-std)[sorted_indices], (mean+std)[sorted_indices], color='blue', alpha=0.25)
    axes[1].plot(oracle_assessment[sorted_indices], leakage_assessment.mean(axis=0)[sorted_indices], linestyle='none', marker='.', color='blue', **PLOT_KWARGS)
    fig.tight_layout()
    fig.savefig(dest, **SAVEFIG_KWARGS)
    plt.close(fig)

def plot_ll_hparam_sweep(logging_dir):
    with open(os.path.join(logging_dir, 'results.pickle'), 'rb') as f:
        results = pickle.load(f)
    hparam_names = ['theta_lr', 'etat_lr', 'starting_prob', 'theta_weight_decay', 'etat_steps_per_theta_step']
    result_names = ['forward_dnn_auc', 'reverse_dnn_auc']
    fig, axes = plt.subplots(len(hparam_names), len(result_names), figsize=(PLOT_WIDTH*len(result_names), PLOT_WIDTH*len(hparam_names)))
    if len(hparam_names) == 1:
        axes = axes[np.newaxis, ...]
    if len(result_names) == 1:
        axes = axes[..., np.newaxis]
    best_reverse_dnn_auc = np.max(results['reverse_dnn_auc'])
    best_forward_dnn_auc = np.inf
    for idx in range(len(results['forward_dnn_auc'])):
        forward_dnn_auc = results['forward_dnn_auc'][idx]
        reverse_dnn_auc = results['reverse_dnn_auc'][idx]
        if reverse_dnn_auc == best_reverse_dnn_auc:
            if forward_dnn_auc < best_forward_dnn_auc:
                best_forward_dnn_auc = forward_dnn_auc
                chosen_settings = {hparam_name: results[hparam_name][idx] for hparam_name in hparam_names}
                chosen_results = {result_name: results[result_name][idx] for result_name in result_names}
    for row_idx, (hparam_name, axes_row) in enumerate(zip(hparam_names, axes)):
        for col_idx, (result_name, ax) in enumerate(zip(result_names, axes_row)):
            hparam_vals = results[hparam_name]
            distinct_hparam_vals = list(set(hparam_vals))
            if all(isinstance(x, int) or isinstance(x, float) for x in hparam_vals):
                distinct_hparam_vals.sort()
            result_vals = results[result_name]
            label_to_num = {hparam_name: idx for idx, hparam_name in enumerate(distinct_hparam_vals)}
            xx = [label_to_num[x] for x in hparam_vals]
            ax.plot(xx, result_vals, color='blue', marker='.', linestyle='none', markersize=1, **PLOT_KWARGS)
            ax.plot([label_to_num[chosen_settings[hparam_name]]], [chosen_results[result_name]], color='red', marker='.', linestyle='none')
            ax.set_xticks(list(label_to_num.values()))
            if hparam_name in ['lr', 'eps', 'weight_decay']:
                ticklabels = [f'{x:.1e}' for x in label_to_num.keys()]
            else:
                ticklabels = [str(x) for x in label_to_num.keys()]
            ax.set_xticklabels(ticklabels, rotation=45 if 'lr' in hparam_name else 0, ha='right')
            ax.set_xlabel(hparam_name.replace('_', '\_'))
            ax.set_ylabel(result_name.replace('_', '\_'))
            if 'loss' in result_name:
                ax.set_yscale('symlog')
    fig.tight_layout()
    fig.savefig(os.path.join(logging_dir, 'hparam_sweep.pdf'), **SAVEFIG_KWARGS)
    plt.close(fig)
    return chosen_settings

def plot_classifiers_hparam_sweep(logging_dir):
    with open(os.path.join(logging_dir, 'results.pickle'), 'rb') as f:
        results = pickle.load(f)
    result_names = ['min_rank', 'final_rank', 'min_loss', 'final_loss']
    assert all(name in results.keys() for name in result_names)
    hparam_names = [key for key in results.keys() if key not in result_names]
    chosen_settings, chosen_results = {}, {}
    best_min_rank = np.min(results['min_rank'])
    best_min_loss = np.inf
    for idx in range(len(results['min_rank'])):
        min_rank = results['min_rank'][idx]
        min_loss = results['min_loss'][idx]
        if min_rank <= 1.01*best_min_rank:
            if min_loss < best_min_loss:
                best_min_loss = min_loss
                chosen_settings = {hparam_name: results[hparam_name][idx] for hparam_name in hparam_names}
                chosen_results = {result_name: results[result_name][idx] for result_name in result_names}
    fig, axes = plt.subplots(len(hparam_names), len(result_names), figsize=(PLOT_WIDTH*len(result_names), PLOT_WIDTH*len(hparam_names)))
    for row_idx, (hparam_name, axes_row) in enumerate(zip(hparam_names, axes)):
        for col_idx, (result_name, ax) in enumerate(zip(result_names, axes_row)):
            hparam_vals = results[hparam_name]
            distinct_hparam_vals = list(set(hparam_vals))
            if all(isinstance(x, int) or isinstance(x, float) for x in hparam_vals):
                distinct_hparam_vals.sort()
            result_vals = results[result_name]
            label_to_num = {hparam_name: idx for idx, hparam_name in enumerate(distinct_hparam_vals)}
            xx = [label_to_num[x] for x in hparam_vals]
            ax.plot(xx, result_vals, color='blue', marker='.', linestyle='none', markersize=1, **PLOT_KWARGS)
            ax.plot([label_to_num[chosen_settings[hparam_name]]], [chosen_results[result_name]], color='red', marker='.', linestyle='none')
            ax.set_xticks(list(label_to_num.values()))
            if hparam_name in ['lr', 'eps', 'weight_decay']:
                ticklabels = [f'{x:.1e}' for x in label_to_num.keys()]
            else:
                ticklabels = [str(x) for x in label_to_num.keys()]
            ax.set_xticklabels(ticklabels, rotation=45 if hparam_name == 'lr' else 0, ha='right')
            ax.set_xlabel(hparam_name.replace('_', '\_'))
            ax.set_ylabel(result_name.replace('_', '\_'))
            if 'loss' in result_name:
                ax.set_yscale('symlog')
    fig.tight_layout()
    fig.savefig(os.path.join(logging_dir, 'hparam_sweep.pdf'), **SAVEFIG_KWARGS)
    plt.close(fig)
    return chosen_settings

def plot_inclusion_probs(inclusion_probs, ax):
    line, *_ = ax.plot(inclusion_probs.squeeze(), color='blue', marker='.', linestyle='-', markersize=1, linewidth=0.1, **PLOT_KWARGS)
    return line

def anim_inclusion_probs_traj(logging_dir):
    log_gammas = extract_log_gamma(logging_dir)
    output_path = os.path.join(logging_dir, 'inclusion_prob_traj.gif')
    fig, axes = plt.subplots(1, 2, figsize=(2*PLOT_WIDTH, PLOT_WIDTH))
    axes[0].set_xlabel(r'Timestep $t$')
    axes[1].set_xlabel(r'Timestep $t$')
    axes[0].set_ylabel(r'Estimated leakage of $X_t$')
    axes[1].set_ylabel(r'Estimated leakage of $X_t$')
    axes[0].set_ylim(0, 1)
    axes[1].set_yscale('log')
    indices = np.linspace(0, len(log_gammas[1])-1, 100).astype(int)
    with imageio.get_writer(output_path, mode='I', fps=20) as writer:
        with tempfile.TemporaryDirectory() as temp_dir:
            for idx in indices:
                step = log_gammas[0][idx]
                log_gamma = log_gammas[1][idx]
                filename = os.path.join(temp_dir, f'step={step}.png')
                x0 = plot_inclusion_probs(np.exp(log_gamma), axes[0])
                x1 = plot_inclusion_probs(np.exp(log_gamma), axes[1])
                fig.suptitle(f'Training step: {step}')
                fig.tight_layout()
                fig.savefig(filename, **SAVEFIG_KWARGS)
                x0.remove()
                x1.remove()
                image = imageio.imread(filename)
                writer.append_data(image)
    plt.close(fig)
    
def plot_vs_reference(logging_dir, inclusion_probs, reference):
    fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_WIDTH))
    ax.set_xlabel('Reference')
    ax.set_ylabel('Gammas')
    ax.plot(reference.squeeze(), inclusion_probs.squeeze(), marker='.', markersize=1, linestyle='none', color='blue')
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(os.path.join(logging_dir, 'comparison_to_reference.png'))
    plt.close(fig)

def plot_training_curves(logging_dir, anim_gammas=True, reference=None):
    training_curves = get_training_curves(logging_dir)
    fig, axes = plt.subplots(1, 4, figsize=(4*PLOT_WIDTH, 1*PLOT_WIDTH))
    axes = axes.flatten()
    if all(x in training_curves for x in ['train_etat_loss', 'val_etat_loss']):
        axes[0].plot(*training_curves['train_etat_loss'], color='blue', linestyle='--', label='train', **PLOT_KWARGS)
        axes[0].plot(*training_curves['val_etat_loss'], color='blue', linestyle='-', label='val', **PLOT_KWARGS)
    if all(x in training_curves for x in ['train_theta_loss', 'val_theta_loss']):
        axes[1].plot(*training_curves['train_theta_loss'], color='red', linestyle='--', label='train', **PLOT_KWARGS)
        axes[1].plot(*training_curves['val_theta_loss'], color='red', linestyle='-', label='val', **PLOT_KWARGS)
    if all(x in training_curves for x in ['train_theta_rank', 'val_theta_rank']):
        axes[2].plot(*training_curves['train_theta_rank'], color='red', linestyle='--', label='train', **PLOT_KWARGS)
        axes[2].plot(*training_curves['val_theta_rank'], color='red', linestyle='-', label='val', **PLOT_KWARGS)
    if 'oracle_snr_corr' in training_curves:
        axes[3].plot(*training_curves['oracle_snr_corr'], color='blue', linestyle='-', **PLOT_KWARGS)
    for ax in axes:
        ax.set_xlabel('Training step')
    axes[0].set_ylabel(r'Loss ($\tilde{\eta}$)')
    axes[1].set_ylabel(r'Loss ($\theta$)')
    axes[2].set_ylabel(r'Rank ($\theta$)')
    axes[3].set_ylabel(r'Spearman correlation with oracle SNR')
    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    axes[0].set_yscale('symlog')
    axes[1].set_yscale('log')
    fig.suptitle('Adversarial leakage localization training curves')
    fig.tight_layout()
    fig.savefig(os.path.join(logging_dir, 'training_curves.png'), **SAVEFIG_KWARGS)
    plt.close(fig)