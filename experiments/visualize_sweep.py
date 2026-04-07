import argparse
from pathlib import Path
from typing import List
from collections import defaultdict
from math import log

import pandas
import numpy as np
from scipy.stats import spearmanr
from matplotlib import pyplot as plt

from leakage_localization.evaluation.mtd import compute_mtd
from leakage_localization.evaluation.oracle_agreement import OracleAgreement
from leakage_localization.datasets import DATASET

from init_things import *
from utils.visualize_runs import *

def load_sweep(sweep_dir: Path) -> pandas.DataFrame:
    trial_dirs: List[Path] = []
    for x in sweep_dir.iterdir():
        if not x.is_dir():
            continue
        if not 'trial_' in x.name:
            continue
        if not (x / 'metrics.csv').exists():
            continue
        trial_dirs.append(x)
    trial_dirs.sort(key=lambda x: x.name)

    data = defaultdict(list)
    for trial_dir in trial_dirs:
        data['path'].append(trial_dir)
        attack_metrics_path = trial_dir / 'attack_metrics.npz'
        assert attack_metrics_path.exists(), attack_metrics_path
        attack_metrics = np.load(attack_metrics_path, allow_pickle=True)
        for metric in ['loss', 'rank', 'acc']:
            data[metric].append(attack_metrics[f'test/{metric}'].item())
            for byte_idx in range(16):
                data[f'{metric}/{byte_idx}'].append(attack_metrics[f'test/{metric}/{byte_idx}'].item())
        data['mtd'].append(attack_metrics['test/mtd'].item())
        for byte_idx in range(16):
            data[f'mtd/{byte_idx}'].append(attack_metrics['per_byte_mtd'][byte_idx])
        for attr_method in ['gradvis', 'input_x_gradient']:
            # fwd DNN occlusion — new format is .npz, old format is .npy
            fwd_dnno_npz = trial_dir / f'fwd_dnno_occl.{attr_method}.npz'
            fwd_dnno_npy = trial_dir / f'fwd_dnno_occl.{attr_method}.npy'
            assert fwd_dnno_npz.exists() or fwd_dnno_npy.exists(), fwd_dnno_npz
            if fwd_dnno_npz.exists():
                fwd_dnno_data = np.load(fwd_dnno_npz, allow_pickle=True)
                data[f'fwd_dnno/{attr_method}'].append(fwd_dnno_data['fwd-dnno-occl'].mean())
                b2_key = 'fwd-dnno-occl/2'
                data[f'fwd_dnno/{attr_method}/2'].append(fwd_dnno_data[b2_key].mean() if b2_key in fwd_dnno_data else np.nan)
            else:
                fwd_dnno = np.load(fwd_dnno_npy)
                data[f'fwd_dnno/{attr_method}'].append(fwd_dnno.mean())
                data[f'fwd_dnno/{attr_method}/2'].append(np.nan)
            # rev DNN occlusion
            rev_dnno_npz = trial_dir / f'rev_dnno_occl.{attr_method}.npz'
            rev_dnno_npy = trial_dir / f'rev_dnno_occl.{attr_method}.npy'
            assert rev_dnno_npz.exists() or rev_dnno_npy.exists(), rev_dnno_npz
            if rev_dnno_npz.exists():
                rev_dnno_data = np.load(rev_dnno_npz, allow_pickle=True)
                data[f'rev_dnno/{attr_method}'].append(rev_dnno_data['rev-dnno-occl'].mean())
                b2_key = 'rev-dnno-occl/2'
                data[f'rev_dnno/{attr_method}/2'].append(rev_dnno_data[b2_key].mean() if b2_key in rev_dnno_data else np.nan)
            else:
                rev_dnno = np.load(rev_dnno_npy)
                data[f'rev_dnno/{attr_method}'].append(rev_dnno.mean())
                data[f'rev_dnno/{attr_method}/2'].append(np.nan)
            # TA MTD — new format has 'ta-mtd' (full-key) and 'ta-mtd/{b}' (per-byte)
            ta_mtd_path = trial_dir / f'ta_mtd.{attr_method}.npz'
            assert ta_mtd_path.exists(), ta_mtd_path
            ta_mtd_data = np.load(ta_mtd_path, allow_pickle=True)
            if 'ta-mtd' in ta_mtd_data:
                data[f'ta_mtd/{attr_method}'].append(float(ta_mtd_data['ta-mtd']))
                for byte_idx in range(16):
                    data[f'ta_mtd/{attr_method}/{byte_idx}'].append(float(ta_mtd_data[f'ta-mtd/{byte_idx}']))
            else:
                # old format: 'mtd' is the per-byte array; no full-key MTD saved
                data[f'ta_mtd/{attr_method}'].append(np.nan)
                for byte_idx in range(16):
                    data[f'ta_mtd/{attr_method}/{byte_idx}'].append(ta_mtd_data['mtd'][byte_idx])
            # white-box agreement
            white_box_path = trial_dir / f'white_box_agreement.{attr_method}.npz'
            assert white_box_path.exists(), white_box_path
            white_box = np.load(white_box_path, allow_pickle=True)
            for byte_idx in range(16):
                data[f'white_box_spearman/{attr_method}/{byte_idx}'].append(white_box['spearman'][byte_idx])
                data[f'white_box_auroc/{attr_method}/{byte_idx}'].append(white_box['auroc'][byte_idx])
            data[f'white_box_spearman/{attr_method}/full'].append(float(white_box['full_spearman']) if 'full_spearman' in white_box else np.nan)
            data[f'white_box_auroc/{attr_method}/full'].append(float(white_box['full_auroc']) if 'full_auroc' in white_box else np.nan)
    data = pandas.DataFrame(data)
    for attr_method in ['gradvis', 'input_x_gradient']:
        data[f'white_box_spearman/{attr_method}'] = data[[f'white_box_spearman/{attr_method}/{byte_idx}' for byte_idx in range(16)]].mean(axis=1)
        data[f'white_box_auroc/{attr_method}'] = data[[f'white_box_auroc/{attr_method}/{byte_idx}' for byte_idx in range(16)]].mean(axis=1)
    data['mean_acc'] = data[[f'acc/{byte_idx}' for byte_idx in range(16)]].mean(axis=1)
    for attr_method in ['gradvis', 'input_x_gradient']:
        data[f'mean_ta_mtd/{attr_method}'] = data[[f'ta_mtd/{attr_method}/{byte_idx}' for byte_idx in range(16)]].mean(axis=1)
    return data

def get_best_attacker(sweep: pandas.DataFrame) -> Path:
    best_row = sweep.loc[sweep['acc'].idxmax()]
    best_path = best_row['path']
    return best_path

def get_best_localizer(sweep: pandas.DataFrame) -> Path:
    best_row = sweep.loc[sweep['white_box_auroc/input_x_gradient'].idxmax()]
    best_path = best_row['path']
    return best_path

# Rows: metrics + methods
# Columns: datasets
def tabulate_best_performance(sweep: pandas.DataFrame, dest: Path):
    rv: List[str] = []
    rv.append(r'\begin{tabular}{c lccc}')
    rv.append(r'\toprule')
    rv.append(r'& ASCADv1 (fixed) & ASCADv1 (variable) & CHES-CTF-2018 \\')

def run_plot_training_curves(sweep: pandas.DataFrame, dest: Path):
    best_attacker_path = get_best_attacker(sweep)
    best_localizer_path = get_best_localizer(sweep)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, WIDTH/2))
    plot_training_curves(
        best_attacker_path, axes[0], 'loss', color='red',
        train_plot_kwargs={'label': 'Best attacker (train)'},
        val_plot_kwargs={'label': 'Best attacker (val)'}
    )
    plot_training_curves(
        best_localizer_path, axes[0], 'loss', color='blue',
        train_plot_kwargs={'label': 'Best localizer (train)'},
        val_plot_kwargs={'label': 'Best localizer (val)'}
    )
    plot_training_curves(
        best_attacker_path, axes[1], 'acc', color='red',
        train_plot_kwargs={'label': 'Best attacker (train)'},
        val_plot_kwargs={'label': 'Best attacker (val)'}
    )
    plot_training_curves(
        best_localizer_path, axes[1], 'acc', color='blue',
        train_plot_kwargs={'label': 'Best localizer (train)'},
        val_plot_kwargs={'label': 'Best localizer (val)'}
    )
    random_loss = log(256)
    axes[0].set_ylim(0, 1.1*random_loss)
    axes[1].set_ylim(0, 1)
    axes[0].set_xlabel('Training step')
    axes[0].set_ylabel(r'Cross-entropy loss $\downarrow$')
    axes[1].set_xlabel('Training step')
    axes[1].set_ylabel(r'Accuracy $\uparrow$')
    axes[0].legend(loc='upper right', framealpha=0., fontsize=6)
    axes[1].legend(loc='lower right', framealpha=0., fontsize=6)
    fig.tight_layout()
    fig.savefig(dest, dpi=DPI)
    plt.close(fig)

def run_plot_mtd(sweep: pandas.DataFrame, dest: Path):
    best_attacker_path = get_best_attacker(sweep)
    best_localizer_path = get_best_localizer(sweep)
    fig, ax = plt.subplots(1, 1, figsize=(WIDTH/2, WIDTH/2))
    plot_mtd(best_attacker_path, ax, color='red', worst_byte_kwargs=dict(label='Best attacker'))
    plot_mtd(best_localizer_path, ax, color='blue', worst_byte_kwargs=dict(label='Best localizer'))
    ax.set_xlabel('Traces seen')
    ax.set_ylabel('Byte rank')
    ax.legend(loc='upper right', framealpha=0, fontsize=6)
    ax.set_xscale('log')
    fig.tight_layout()
    fig.savefig(dest, dpi=DPI)
    plt.close(fig)

def run_white_box_agreement(sweep: pandas.DataFrame, dest: Path):
    best_attacker_path = get_best_attacker(sweep)
    best_localizer_path = get_best_localizer(sweep)
    best_attacker_inputxgrad = np.load(best_attacker_path / 'input_x_gradient.npy')[2, :]
    best_localizer_inputxgrad = np.load(best_localizer_path / 'input_x_gradient.npy')[2, :]
    fig, axes = plt.subplots(3, 1, figsize=(WIDTH, WIDTH))
    axes[0].plot(best_attacker_inputxgrad, color='blue')
    axes[1].plot(best_localizer_inputxgrad, color='blue')
    ref = plot_ascadv1_oracle_leakiness(dest.parent.parent / '..' / 'snr', axes[2])
    best_attacker_agreement = spearmanr(best_attacker_inputxgrad, ref).statistic
    best_localizer_agreement = spearmanr(best_localizer_inputxgrad, ref).statistic
    axes[0].text(0.02, 0.98, f'Spearman r = {best_attacker_agreement:.3f}',
                 transform=axes[0].transAxes, va='top', ha='left')
    axes[1].text(0.02, 0.98, f'Spearman r = {best_localizer_agreement:.3f}',
                 transform=axes[1].transAxes, va='top', ha='left')
    axes[0].set_title('Best attacker input*grad')
    axes[1].set_title('Best localizer input*grad')
    axes[2].set_title('White box SNR')
    axes[0].set_xlabel(r'Time $t$')
    axes[1].set_xlabel(r'Time $t$')
    axes[2].set_xlabel(r'Time $t$')
    axes[0].set_ylabel(r'Estimated leakiness of $X_t$')
    axes[1].set_ylabel(r'Estimated leakiness of $X_t$')
    axes[2].set_ylabel(r'Estimated leakiness of $X_t$')
    fig.tight_layout()
    fig.savefig(dest, dpi=DPI)
    plt.close(fig)

def run_minimal_white_box_agreement(sweep: pandas.DataFrame, dest: Path):
    best_localizer_path = get_best_localizer(sweep)
    best_localizer_inputxgrad = np.load(best_localizer_path / 'input_x_gradient.npy')[2, :]
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, WIDTH))
    axes[0].plot(best_localizer_inputxgrad, color='blue')
    plot_ascadv1_oracle_leakiness(dest.parent.parent / '..' / 'snr', axes[1])
    axes[0].set_title('Black box (ours)')
    axes[1].set_title('White box (ground truth)')
    axes[0].set_xlabel(r'Time $t$')
    axes[1].set_xlabel(r'Time $t$')
    axes[0].set_ylabel(r'Estimated leakiness of $X_t$')
    axes[1].set_ylabel(r'Estimated leakiness of $X_t$')
    axes[1].legend(loc='upper right', ncol=3, framealpha=0, fontsize=6, title='Leaky intermediate variables', title_fontsize=8)
    fig.tight_layout()
    fig.savefig(dest, dpi=DPI)
    plt.close(fig)

def run_white_box_agreement_all_bytes(
        sweep: pandas.DataFrame,
        dest: Path,
        dataset: DATASET = 'ascadv1-fixed',
        auroc_percentile: float = 0.9999,
        snr_threshold: Optional[float] = None,
):
    best_attacker_path = get_best_attacker(sweep)
    best_localizer_path = get_best_localizer(sweep)

    best_attacker_inputxgrad = np.load(best_attacker_path / 'input_x_gradient.npy')  # [16, T]
    best_localizer_inputxgrad = np.load(best_localizer_path / 'input_x_gradient.npy')  # [16, T]

    attacker_wb = np.load(best_attacker_path / 'white_box_agreement.input_x_gradient.npz', allow_pickle=True)
    localizer_wb = np.load(best_localizer_path / 'white_box_agreement.input_x_gradient.npz', allow_pickle=True)

    snr_dir = dest.parent.parent / '..' / 'snr'
    oracle = OracleAgreement(snr_dir, dataset)
    # oracle_leakiness[b] uses the correct per-byte variable set (e.g. only 'subbytes'
    # for bytes 0-1 in ASCADv1, all masked vars for bytes 2-15).
    oracle_leakiness = oracle.oracle_leakiness          # [16, T]
    binary_labels = oracle.get_binary_labels('attack', auroc_percentile, snr_threshold)  # [16, T] bool

    # Diagnostic: per-variable label counts, to identify which variable drives dense labels
    _threshold = oracle.get_threshold('attack', auroc_percentile, snr_threshold)
    print(f'\nBinary label threshold (SNR): {_threshold:.4f}')
    print(f'{"byte":>4}  {"variable":<30}  {"n_leaky":>8}  {"frac":>6}')
    for byte_idx, var_names in oracle.variables.items():
        for var_name in var_names:
            snr = np.load(snr_dir / f'{var_name}.attack.npy')
            row = min(byte_idx, snr.shape[0] - 1)
            n_leaky = int((snr[row] > _threshold).sum())
            frac = n_leaky / snr.shape[1]
            if n_leaky > 0:
                print(f'{byte_idx:>4}  {var_name:<30}  {n_leaky:>8}  {frac:>6.3f}')

    t = oracle_leakiness.shape[1]
    xs = np.arange(t)

    with plt.rc_context({'font.size': 5, 'axes.labelsize': 5, 'xtick.labelsize': 4, 'ytick.labelsize': 4, 'axes.titlesize': 4}):
        fig, axes = plt.subplots(3, 16, figsize=(WIDTH * 4, WIDTH * 3 / 4))

        for byte_idx in range(16):
            attacker_spearman  = attacker_wb['spearman'][byte_idx]
            attacker_auroc     = attacker_wb['auroc'][byte_idx]
            localizer_spearman = localizer_wb['spearman'][byte_idx]
            localizer_auroc    = localizer_wb['auroc'][byte_idx]
            labels = binary_labels[byte_idx]

            axes[0, byte_idx].plot(best_attacker_inputxgrad[byte_idx], color='blue', lw=0.5)
            axes[1, byte_idx].plot(best_localizer_inputxgrad[byte_idx], color='blue', lw=0.5)
            axes[2, byte_idx].plot(oracle_leakiness[byte_idx], color='black', lw=0.5)

            # Shade leaky timesteps (binary AUROC labels) on all three rows
            for row in range(3):
                axes[row, byte_idx].fill_between(
                    xs, 0, 1,
                    where=labels,
                    transform=axes[row, byte_idx].get_xaxis_transform(),
                    color='red', alpha=0.25, linewidth=0,
                )

            axes[0, byte_idx].set_title(
                f'Byte {byte_idx}\n$\\rho$={attacker_spearman:.2f}, AUC={attacker_auroc:.2f}'
            )
            axes[1, byte_idx].set_title(
                f'$\\rho$={localizer_spearman:.2f}, AUC={localizer_auroc:.2f}'
            )
            axes[2, byte_idx].set_title(
                f'n_leaky={labels.sum()}'
            )

            for row in range(3):
                axes[row, byte_idx].set_xticks([])
                axes[row, byte_idx].set_yticks([])

        axes[0, 0].set_ylabel('Attacker\nInput×Grad')
        axes[1, 0].set_ylabel('Localizer\nInput×Grad')
        axes[2, 0].set_ylabel('Oracle SNR\n(composite)')

        fig.tight_layout()
        fig.savefig(dest, dpi=DPI, bbox_inches='tight')
        plt.close(fig)

def run_plot_oracle_snr_histograms(
        dest: Path,
        dataset: DATASET = 'ascadv1-fixed',
        auroc_percentile: float = 0.9999,
        snr_threshold: Optional[float] = None,
        n_bins: int = 100,
):
    snr_dir = dest.parent.parent / '..' / 'snr'
    oracle = OracleAgreement(snr_dir, dataset)
    threshold = oracle.get_threshold('attack', auroc_percentile, snr_threshold)
    null_mean = (oracle.num_classes - 1) / (oracle.n_traces['attack'] - oracle.num_classes)

    # Compute max SNR across variables per timestep per byte.
    # This is the quantity that directly determines binary labeling:
    # a timestep is leaky iff max_var(SNR) > threshold.
    feature_count = oracle.feature_count
    max_snr = np.zeros((oracle.byte_count, feature_count), dtype=np.float32)
    for byte_idx, var_names in oracle.variables.items():
        for var_name in var_names:
            snr = np.load(snr_dir / f'{var_name}.attack.npy')
            row = min(byte_idx, snr.shape[0] - 1)
            max_snr[byte_idx] = np.maximum(max_snr[byte_idx], snr[row])

    all_pos = max_snr[max_snr > 0]
    bins = np.logspace(np.log10(float(all_pos.min())), np.log10(float(max_snr.max())), n_bins + 1)

    threshold_label = f'threshold={threshold:.4f}' + ('' if snr_threshold is not None else f' (F-null p={auroc_percentile})')
    with plt.rc_context({'font.size': 6, 'axes.labelsize': 6, 'xtick.labelsize': 5, 'ytick.labelsize': 5, 'axes.titlesize': 6}):
        fig, axes = plt.subplots(4, 4, figsize=(WIDTH, WIDTH))
        for byte_idx, ax in enumerate(axes.flat):
            snr = max_snr[byte_idx]
            ax.hist(snr[snr > 0], bins=bins, color='steelblue', edgecolor='none')
            ax.axvline(threshold, color='red',    lw=1.0, linestyle='--', label=threshold_label)
            ax.axvline(null_mean,  color='orange', lw=0.8, linestyle=':',  label='Null mean')
            ax.set_xscale('log')
            ax.set_yscale('log')
            n_leaky = int((snr > threshold).sum())
            ax.set_title(f'Byte {byte_idx}  (n_leaky={n_leaky})')
            ax.set_xlabel('Max SNR across variables')
            ax.set_ylabel('Count')
        axes.flat[0].legend(fontsize=4, loc='upper right')
        fig.tight_layout()
        fig.savefig(dest, dpi=DPI, bbox_inches='tight')
        plt.close(fig)

def run_plot_perbyte_pervariable_snr_histograms(
        dest: Path,
        dataset: DATASET = 'ascadv1-fixed',
        auroc_percentile: float = 0.9999,
        snr_threshold: Optional[float] = None,
        n_bins: int = 100,
):
    """One column per byte, one row per variable, showing raw SNR histograms."""
    snr_dir = dest.parent.parent / '..' / 'snr'
    oracle = OracleAgreement(snr_dir, dataset)
    threshold = oracle.get_threshold('attack', auroc_percentile, snr_threshold)

    # Collect variable names across all bytes (preserve order)
    all_vars = []
    for var_names in oracle.variables.values():
        for v in var_names:
            if v not in all_vars:
                all_vars.append(v)

    # Compute shared log-spaced bins across all SNR files
    all_snr_vals = []
    for var_name in all_vars:
        snr_file = np.load(snr_dir / f'{var_name}.attack.npy')
        all_snr_vals.append(snr_file[snr_file > 0].ravel())
    all_snr_cat = np.concatenate(all_snr_vals)
    bins = np.logspace(np.log10(float(all_snr_cat.min())), np.log10(float(all_snr_cat.max())), n_bins + 1)

    n_vars = len(all_vars)
    with plt.rc_context({'font.size': 4, 'axes.labelsize': 4, 'xtick.labelsize': 3, 'ytick.labelsize': 3, 'axes.titlesize': 4}):
        fig, axes = plt.subplots(n_vars, 16, figsize=(WIDTH * 4, WIDTH * n_vars / 4))
        for row_idx, var_name in enumerate(all_vars):
            snr_path = snr_dir / f'{var_name}.attack.npy'
            snr_file = np.load(snr_path)
            for byte_idx in range(16):
                ax = axes[row_idx, byte_idx]
                var_names_for_byte = oracle.variables.get(byte_idx, [])
                if var_name not in var_names_for_byte:
                    ax.set_visible(False)
                    continue
                row = min(byte_idx, snr_file.shape[0] - 1)
                snr = snr_file[row]
                ax.hist(snr[snr > 0], bins=bins, color='steelblue', edgecolor='none')
                ax.axvline(threshold, color='red', lw=0.8, linestyle='--')
                ax.set_xscale('log')
                ax.set_yscale('log')
                n_leaky = int((snr > threshold).sum())
                if row_idx == 0:
                    ax.set_title(f'B{byte_idx}')
                if byte_idx == 0:
                    ax.set_ylabel(var_name.replace('__xor__', '⊕') + f'\nn={n_leaky}', fontsize=3)
                else:
                    ax.set_title(f'n={n_leaky}', fontsize=3)
        fig.tight_layout()
        fig.savefig(dest, dpi=DPI, bbox_inches='tight')
        plt.close(fig)

def run_plot_gradvis_vs_inputxgrad(sweep: pandas.DataFrame, dest: Path):
    with plt.rc_context({'font.size': 6, 'axes.labelsize': 6, 'xtick.labelsize': 5, 'ytick.labelsize': 5}):
        fig, axes = plt.subplots(1, 4, figsize=(WIDTH, WIDTH/4))
        kwargs = dict(
            color='blue',
            marker='.',
            linestyle='none',
            markersize=3
        )
        axes[0].plot(sweep['white_box_auroc/gradvis/full'], sweep['white_box_auroc/input_x_gradient/full'], **kwargs)
        axes[1].plot(sweep['fwd_dnno/gradvis'], sweep['fwd_dnno/input_x_gradient'], **kwargs)
        axes[2].plot(sweep['rev_dnno/gradvis'], sweep['rev_dnno/input_x_gradient'], **kwargs)
        axes[3].plot(sweep['ta_mtd/gradvis'], sweep['ta_mtd/input_x_gradient'], **kwargs)
        add_dline(axes[0], color='grey', linestyle=':')
        add_dline(axes[1], color='grey', linestyle=':')
        add_dline(axes[2], color='grey', linestyle=':')
        add_dline(axes[3], color='grey', linestyle=':')
        axes[0].set_xlabel('GradVis')
        axes[0].set_ylabel('Input * Grad')
        axes[0].set_title(r'White box AUROC $\uparrow$')
        axes[1].set_xlabel('GradVis')
        axes[1].set_ylabel('Input * Grad')
        axes[1].set_title(r'Forward DNN occlusion $\downarrow$')
        axes[2].set_xlabel('GradVis')
        axes[2].set_ylabel('Input * Grad')
        axes[2].set_title(r'Reverse DNN occlusion $\uparrow$')
        axes[3].set_xlabel('GradVis')
        axes[3].set_ylabel('Input * Grad')
        axes[3].set_title(r'Template attack MTD $\downarrow$')
        for ax in axes:
            ax.tick_params(axis='both', which='both', pad=2)
        fig.tight_layout()
        fig.savefig(dest, dpi=DPI)
        plt.close(fig)

def run_plot_perbyte_attack_vs_loc(sweep: pandas.DataFrame, dest: Path, byte: int = 2):
    with plt.rc_context({'font.size': 6, 'axes.labelsize': 6, 'xtick.labelsize': 5, 'ytick.labelsize': 5}):
        fig, axes = plt.subplots(1, 4, figsize=(WIDTH, WIDTH/4))
        kwargs = dict(
            color='blue',
            marker='.',
            linestyle='none',
            markersize=3
        )
        axes[0].plot(sweep[f'acc/{byte}'], sweep[f'white_box_auroc/gradvis/{byte}'], **kwargs)
        axes[1].plot(sweep[f'acc/{byte}'], sweep[f'fwd_dnno/gradvis'], **kwargs)
        axes[2].plot(sweep[f'acc/{byte}'], sweep[f'rev_dnno/gradvis'], **kwargs)
        axes[3].plot(sweep[f'acc/{byte}'], sweep[f'ta_mtd/gradvis/{byte}'], **kwargs)
        axes[0].set_xlabel('Accuracy')
        axes[1].set_xlabel('Accuracy')
        axes[2].set_xlabel('Accuracy')
        axes[3].set_xlabel('Accuracy')
        axes[0].set_ylabel('White box AUROC')
        axes[1].set_ylabel('Forward DNN occlusion')
        axes[2].set_ylabel('Reverse DNN occlusion')
        axes[3].set_ylabel('Tempalate attack MTD')
        axes[0].set_xscale('log')
        axes[1].set_xscale('log')
        axes[2].set_xscale('log')
        axes[3].set_xscale('log')
        axes[3].set_yscale('log')
        for byte_idx in range(16):
            print(f'Best white box AUROC (byte {byte_idx}): {sweep[f"white_box_auroc/gradvis/{byte_idx}"].max()}')
        for ax in axes:
            ax.tick_params(axis='both', which='both', pad=2)
        fig.tight_layout()
        fig.savefig(dest, dpi=DPI)
        plt.close(fig)
    
def run_plot_attack_vs_loc(sweep: pandas.DataFrame, dest: Path):
    with plt.rc_context({'font.size': 6, 'axes.labelsize': 6, 'xtick.labelsize': 5, 'ytick.labelsize': 5}):
        fig, axes = plt.subplots(1, 4, figsize=(WIDTH, WIDTH/4))
        kwargs = dict(
            color='blue',
            marker='.',
            linestyle='none',
            markersize=3
        )
        axes[0].plot(sweep['mean_acc'], sweep['white_box_auroc/gradvis/full'], **kwargs)
        axes[1].plot(sweep['mean_acc'], sweep['fwd_dnno/gradvis'], **kwargs)
        axes[2].plot(sweep['mean_acc'], sweep['rev_dnno/gradvis'], **kwargs)
        axes[3].plot(sweep['mean_acc'], sweep['mean_ta_mtd/gradvis'], **kwargs)
        axes[0].set_xlabel('Mean per-byte accuracy')
        axes[1].set_xlabel('Mean per-byte accuracy')
        axes[2].set_xlabel('Mean per-byte accuracy')
        axes[3].set_xlabel('Mean per-byte accuracy')
        axes[0].set_ylabel('White box AUROC')
        axes[1].set_ylabel('Forward DNN occlusion')
        axes[2].set_ylabel('Reverse DNN occlusion')
        axes[3].set_ylabel('Tempalate attack MTD')
        axes[0].set_xscale('log')
        axes[1].set_xscale('log')
        axes[2].set_xscale('log')
        axes[3].set_xscale('log')
        axes[3].set_yscale('log')
        for ax in axes:
            ax.tick_params(axis='both', which='both', pad=2)
        fig.tight_layout()
        fig.savefig(dest, dpi=DPI)
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--sweep-dir', type=Path, required=True,
        help='Base directory of the sweep to be plotted.'
    )
    parser.add_argument(
        '--dest', type=Path, default=None,
        help='Directory in which to save figures. Defaults to a directory called `plots` in the sweep directory.'
    )
    parser.add_argument(
        '--snr-threshold', type=float, default=None,
        help='Override the F-null SNR threshold with a fixed value for binary leakage labels.'
    )
    args = parser.parse_args()

    sweep_dir: Path = args.sweep_dir
    assert isinstance(sweep_dir, Path) and sweep_dir.exists()
    dest: Optional[Path] = args.dest
    if dest is None:
        dest = sweep_dir / 'plots'
        dest.mkdir(exist_ok=True)
    assert isinstance(dest, Path) and dest.exists()
    snr_threshold: Optional[float] = args.snr_threshold

    sweep = load_sweep(sweep_dir)
    print(sweep)
    for col in sweep.columns:
        print(f'\t{col}: {sweep[col].isna().sum()/len(sweep[col])}')
    print(f'Best attacker path: {get_best_attacker(sweep)}')
    print(f'Best localizer path: {get_best_localizer(sweep)}')

    # table listing performance of the best attacker and localizer models

    # training curves for the best attacker and best localizer
    run_plot_training_curves(sweep, dest / 'training_curves.pdf')

    # rank over time for the best attacker and best localizer
    run_plot_mtd(sweep, dest / 'mtd.pdf')

    # leakiness over time visualizations for oracle, best attacker, best localizer
    run_white_box_agreement(sweep, dest / 'white_box_agreement.pdf')
    run_white_box_agreement_all_bytes(sweep, dest / 'white_box_agreement_all_bytes.pdf', snr_threshold=snr_threshold)
    run_minimal_white_box_agreement(sweep, dest / 'minimal_white_box_agreement.pdf')

    # oracle SNR histograms (composite per byte, and per-variable breakdown)
    run_plot_oracle_snr_histograms(dest / 'oracle_snr_histograms.pdf', snr_threshold=snr_threshold)
    run_plot_perbyte_pervariable_snr_histograms(dest / 'oracle_snr_histograms_per_variable.pdf', snr_threshold=snr_threshold)

    # visualizations of the DNN occlusion tests for the oracle, random, best attacker, best localizer

    # visualizations of template attack MTD for the oracle, random, best attacker, best localizer

    # scatterplots showing relationship between the different attack/localization performance metrics
    run_plot_attack_vs_loc(sweep, dest / 'attack_vs_loc.pdf')
    run_plot_perbyte_attack_vs_loc(sweep, dest / 'perbyte_attack_vs_loc.pdf')

    # scatterplots showing relationship between GradVis and input x grad
    run_plot_gradvis_vs_inputxgrad(sweep, dest / 'gradvis_vs_inpxgrad.pdf')

if __name__ == '__main__':
    main()