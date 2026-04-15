from pathlib import Path
from typing import Optional, Dict, Any, List, get_args

import colorcet
import pandas
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from scipy.stats import spearmanr

from leakage_localization.datasets import PARTITION
from leakage_localization.datasets.ascadv1 import repr_target as ascadv1_repr_target

def add_dline(
        ax: Axes,
        **kwargs
):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    low = max(xmin, ymin)
    high = min(xmax, ymax)
    ax.plot((low, high), (low, high), **kwargs)

def plot_ascadv1_oracle_leakiness(
        snr_dir: Path,
        ax: Axes,
        byte: int = 2,
        markers: bool = True,
        arb_byte: bool = False,
):
    marker_kwargs = dict(marker='.', markersize=1) if markers else {}
    byte_label = 'i' if arb_byte else f'{byte}'

    # Bytes 0 and 1 have no masking — use subbytes SNR directly as a single curve.
    if byte in (0, 1):
        snr = np.load(snr_dir / 'subbytes.attack.npy')[byte, :]
        kwargs = dict(color='grey', linestyle='-', label=ascadv1_repr_target('subbytes', byte=byte_label))
        ax.plot(snr, rasterized=True, linewidth=0.3, **marker_kwargs, **kwargs)
        return dict(composite=snr, subbytes=snr)

    int_var_snrs = dict(
        prin = np.load(snr_dir / 'p__xor__k__xor__r_in.attack.npy')[byte, :],
        pr = np.load(snr_dir / 'p__xor__k__xor__r.profile.npy')[byte, :],
        rin = np.load(snr_dir / 'r_in.attack.npy')[0, :],
        rout = np.load(snr_dir / 'r_out.attack.npy')[0, :],
        r = np.load(snr_dir / 'r.attack.npy')[byte, :],
        yrout = np.load(snr_dir / 'subbytes__xor__r_out.attack.npy')[byte, :],
        yr = np.load(snr_dir / 'subbytes__xor__r.attack.npy')[byte, :],
    )
    int_var_kwargs = dict(
        prin = dict(color = 'red', linestyle='--', label=ascadv1_repr_target('p__xor__k__xor__r_in', byte=byte_label)),
        pr = dict(color = 'green', linestyle='--', label=ascadv1_repr_target('p__xor__k__xor__r', byte=byte_label)),
        rin = dict(color='purple', linestyle='-', label=ascadv1_repr_target('r_in', byte=byte_label)),
        rout = dict(color='teal', linestyle='-', label=ascadv1_repr_target('r_out', byte=byte_label)),
        r = dict(color='orange', linestyle='-', label=ascadv1_repr_target('r', byte=byte_label)),
        yrout = dict(color = 'magenta', linestyle='-', label=ascadv1_repr_target('subbytes__xor__r_out', byte=byte_label)),
        yr = dict(color = 'black', linestyle='-', label=ascadv1_repr_target('subbytes__xor__r', byte=byte_label))
    )
    for int_var_name in int_var_snrs.keys():
        int_var_snr = int_var_snrs[int_var_name]
        if int_var_snr is None:
            continue
        kwargs = int_var_kwargs[int_var_name]
        ax.plot(int_var_snr, rasterized=True, linewidth=0.3, **marker_kwargs, **kwargs)
    white_box_composite = np.stack(list(int_var_snrs.values())).mean(axis=0)
    return dict(composite=white_box_composite, **int_var_snrs)

def plot_leakiness_over_time(
        attr: np.ndarray,
        ax: Axes,
        title: Optional[str] = None,
        per_byte_alpha: float = 0.35,
        per_byte_lw: float = 0.4,
        sum_lw: float = 1.2,
        sum_color: str = 'black',
        byte_color: str = 'royalblue',
):
    """Line plot of attribution values over time.

    Draws one thin faded line per byte and a thicker line for the per-timestep
    sum across bytes.

    Args:
        attr: shape (byte_count, feature_count)
    """
    byte_count, feature_count = attr.shape
    timesteps = np.arange(feature_count)
    for byte_idx in range(byte_count):
        ax.plot(timesteps, attr[byte_idx], color=byte_color,
                linewidth=per_byte_lw, alpha=per_byte_alpha, rasterized=True)
    ax.plot(timesteps, attr.sum(axis=0), color=sum_color,
            linewidth=sum_lw, label='sum', rasterized=True)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Estimated leakiness')
    if title is not None:
        ax.set_title(title)


def plot_wb_scatterplots(
        attr: np.ndarray,
        oracle: np.ndarray,
        axes,
        subsample: int = 5,
        color: str = 'royalblue',
):
    """4×4 grid of per-byte estimated-vs-oracle scatterplots.

    Args:
        attr:   (byte_count, feature_count) estimated leakiness
        oracle: (byte_count, feature_count) oracle (white-box) leakiness
        axes:   array of Axes with shape (4, 4) or flat length >= byte_count
        subsample: keep every nth point to limit file size
    """
    flat_axes = np.array(axes).flatten()
    byte_count = attr.shape[0]
    for byte_idx in range(byte_count):
        ax = flat_axes[byte_idx]
        x = oracle[byte_idx, ::subsample]
        y = attr[byte_idx, ::subsample]
        ax.scatter(x, y, s=0.5, alpha=0.3, color=color, rasterized=True, linewidths=0)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f'byte {byte_idx}', fontsize=7)
        ax.set_xlabel('Oracle leakiness', fontsize=6)
        ax.set_ylabel('Estimated', fontsize=6)
        ax.tick_params(labelsize=5)


def plot_wb_comparison_grid(
        attr: np.ndarray,
        oracle: np.ndarray,
        axes,
):
    """2×byte_count grid: top row = estimated per byte, bottom row = oracle per byte.

    Args:
        attr:   (byte_count, feature_count)
        oracle: (byte_count, feature_count)
        axes:   array of Axes with shape (2, byte_count)
    """
    byte_count, feature_count = attr.shape
    timesteps = np.arange(feature_count)
    for byte_idx in range(byte_count):
        ax_est = axes[0, byte_idx]
        ax_ora = axes[1, byte_idx]
        ax_est.plot(timesteps, attr[byte_idx], color='royalblue',
                    linewidth=0.4, rasterized=True)
        ax_est.set_title(f'byte {byte_idx}', fontsize=6)
        ax_est.tick_params(labelsize=5)
        if byte_idx == 0:
            ax_est.set_ylabel('Estimated', fontsize=6)
        ax_ora.plot(timesteps, oracle[byte_idx], color='darkorange',
                    linewidth=0.4, rasterized=True)
        ax_ora.tick_params(labelsize=5)
        if byte_idx == 0:
            ax_ora.set_ylabel('Oracle', fontsize=6)
        ax_ora.set_xlabel('Timestep', fontsize=6)


def plot_rank_trajectories(
        rank_over_time: np.ndarray,
        ax: Axes,
        color: str = 'blue',
        label: Optional[str] = None,
        per_byte_alpha: float = 0.3,
        worst_case_alpha: float = 0.9,
):
    """Plot per-byte rank trajectories plus the worst-case (max) envelope.

    Args:
        rank_over_time: shape (byte_count, trace_count)
    """
    byte_count, trace_count = rank_over_time.shape
    traces_seen = np.arange(1, trace_count + 1)
    for byte_idx in range(byte_count):
        ax.plot(traces_seen, rank_over_time[byte_idx], color=color,
                linewidth=0.3, alpha=per_byte_alpha, rasterized=True)
    ax.plot(traces_seen, rank_over_time.max(axis=0), color=color,
            linewidth=1.2, alpha=worst_case_alpha, label=label, rasterized=True)
    ax.set_xlabel('Traces seen')
    ax.set_ylabel('Rank')


def plot_per_byte_bar(
        values: np.ndarray,
        ax: Axes,
        color: str = 'steelblue',
        label: Optional[str] = None,
        **bar_kwargs
):
    """Bar chart of a (byte_count,) array, one bar per byte."""
    byte_count = len(values)
    ax.bar(np.arange(byte_count), values, color=color, label=label, **bar_kwargs)
    ax.set_xlabel('Byte index')
    ax.set_xticks(np.arange(byte_count))

def plot_mtd(
        run_path: Path,
        ax: Axes,
        worst_byte_kwargs: Optional[Dict[str, Any]] = None,
        other_byte_kwargs: Optional[Dict[str, Any]] = None,
        **common_kwargs
):
    worst_byte_kwargs = worst_byte_kwargs or dict()
    other_byte_kwargs = other_byte_kwargs or dict()
    attack_metrics = np.load(run_path / 'attack_metrics.npz', allow_pickle=True)
    rank_over_time = attack_metrics['rank_over_time']
    byte_count = rank_over_time.shape[0]
    traces_seen = np.arange(1, rank_over_time.shape[1] + 1)
    _worst_byte_kwargs = dict(
        linewidth=2,
    )
    _worst_byte_kwargs.update(common_kwargs)
    _worst_byte_kwargs.update(worst_byte_kwargs)
    _other_byte_kwargs = dict(
        linewidth=0.5,
        alpha=0.5
    )
    _other_byte_kwargs.update(common_kwargs)
    _other_byte_kwargs.update(other_byte_kwargs)
    ax.plot(traces_seen, rank_over_time.max(axis=0), **_worst_byte_kwargs)
    for byte_idx in range(byte_count):
        ax.plot(traces_seen, rank_over_time[byte_idx, :], **_other_byte_kwargs)

def plot_training_curves(
        run_path: Path,
        ax: Axes,
        metric_key: str,
        train_plot_kwargs: Optional[Dict[str, Any]] = None,
        val_plot_kwargs: Optional[Dict[str, Any]] = None,
        **common_plot_kwargs
):
    train_plot_kwargs = train_plot_kwargs or dict()
    val_plot_kwargs = val_plot_kwargs or dict()
    metrics = pandas.read_csv(run_path / 'metrics.csv')
    train_mask = ~metrics['train/loss'].isna()
    val_mask = ~metrics['val/loss'].isna()
    train_steps = metrics['step'][train_mask]
    val_steps = metrics['step'][val_mask]
    train_metric = metrics[f'train/{metric_key}'][train_mask]
    val_metric = metrics[f'val/{metric_key}'][val_mask]
    _train_plot_kwargs = dict(
        color='blue',
        linestyle=':',
        label='train',
        rasterized=True
    )
    _train_plot_kwargs.update(common_plot_kwargs)
    _train_plot_kwargs.update(train_plot_kwargs)
    _val_plot_kwargs = dict(
        color='blue',
        linestyle='-',
        label='val',
        rasterized=True
    )
    _val_plot_kwargs.update(common_plot_kwargs)
    _val_plot_kwargs.update(val_plot_kwargs)
    ax.plot(train_steps, train_metric, **_train_plot_kwargs)
    ax.plot(val_steps, val_metric, **_val_plot_kwargs)

def plot_occlusion_test(
        occlusion_trace_path: Path,
        ax: Axes,
        features_per_trace: int = 1,
        **plot_kwargs
):
    occlusion_trace = np.load(occlusion_trace_path)
    occluded_features = np.linspace(0, features_per_trace, len(occlusion_trace)+1)[1:]
    _plot_kwargs = dict(
        color='blue',
        linestyle='-',
        linewidth=0.2,
        rasterized=True,
        marker='.',
        markersize=3
    )
    _plot_kwargs.update(plot_kwargs)
    ax.plot(occluded_features, occlusion_trace, **_plot_kwargs)

def plot_template_attack_test(
        src: Path,
        ax: Axes,
        **plot_kwargs
):
    data = np.load(src, allow_pickle=True)
    mtd = data['mtd']
    rank_over_time = data['rank_over_time']
    byte_count, trace_count = rank_over_time.shape
    traces_seen = np.arange(1, trace_count + 1)
    _plot_kwargs = dict(
        color='blue',
        rasterized=True
    )
    _plot_kwargs.update(plot_kwargs)
    for byte_idx in range(byte_count):
        ax.plot(traces_seen, rank_over_time[byte_idx, :], linestyle=':', linewidth=0.2, **_plot_kwargs)
    ax.plot(traces_seen, rank_over_time.max(axis=0), linestyle='-', alpha=0.5, **_plot_kwargs)

def plot_white_box_agreement(
        black_box_src: Path,
        white_box_src: Path,
        white_box_illus_ax: Axes,
        oracle_agreement_ax: Axes,
        var_axes: Axes,
):
    black_box_leakiness = np.load(black_box_src)
    white_box_leakiness = {partition: dict() for partition in get_args(PARTITION)}
    for partition in get_args(PARTITION):
        for file in white_box_src.iterdir():
            if not file.name.endswith('.npy'):
                continue
            var_name, partition_name, _ = file.name.split('.')
            if not partition_name == partition:
                continue
            if not var_name in {'p__xor__k__xor__r_in', 'r_in', 'subbytes__xor__r_out', 'r_out', 'subbytes__xor__r', 'r'}:
                continue
            var_leakiness = np.load(file)
            white_box_leakiness[partition][var_name] = var_leakiness
    for var_name, var_leakiness in white_box_leakiness['attack'].items():
        if len(var_leakiness) == 16:
            var_leakiness = var_leakiness[2, :]
        elif len(var_leakiness) == 1:
            var_leakiness = var_leakiness[0, :]
        else:
            assert False
        white_box_illus_ax.plot(var_leakiness, label=var_name, linestyle='-', linewidth=0.2, rasterized=True)
    white_box_assessment = sum(white_box_leakiness['attack'].values())
    oracle_assessment = sum(white_box_leakiness['profile'].values())
    for byte_idx in range(16):
        print(f'Byte idx: {byte_idx}')
        print(f'\tAgreement between profile + attack oracle: {spearmanr(white_box_assessment[byte_idx, :], oracle_assessment[byte_idx, :]).statistic}')
        print(f'\tAgreement between black box + attack oracle: {spearmanr(white_box_assessment[byte_idx, :], black_box_leakiness[byte_idx, :]).statistic}')
    #oracle_agreement_ax.plot(white_box_assessment, black_box_assessment, color='red', marker='.', linestyle='none', markersize=1, alpha=0.1, rasterized=True)
    #oracle_agreement_ax.plot(white_box_assessment, oracle_assessment, color='blue', marker='.', linestyle='none', markersize=1, alpha=0.1, rasterized=True)
    oracle_agreement_ax.plot(white_box_assessment[2, :], black_box_leakiness[2, :], color='blue', linestyle='none', marker='.', markersize=1, rasterized=True)
    for (var_name, var_leakiness), var_ax in zip(white_box_leakiness['attack'].items(), var_axes.flatten()):
        var_ax.plot(var_leakiness[2, :] if len(var_leakiness) > 1 else var_leakiness[0, :], black_box_leakiness[2, :], color='blue', marker='.', linestyle='none', markersize=1, alpha=0.1, rasterized=True)
        var_ax.set_xscale('log')
        var_ax.set_yscale('log')
        var_ax.set_title(var_name, fontsize=6)