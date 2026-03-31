from pathlib import Path
from typing import Optional, Dict, Any, get_args

import pandas
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from scipy.stats import spearmanr

from leakage_localization.datasets import PARTITION

def plot_training_curves(
        run_path: Path,
        ax: Axes,
        metric_key: str,
        train_plot_kwargs: Optional[Dict[str, Any]] = None,
        val_plot_kwargs: Optional[Dict[str, Any]] = None,
        **common_plot_kwargs
):
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