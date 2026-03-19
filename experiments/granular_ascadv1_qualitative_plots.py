import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from matplotlib import pyplot as plt

from matplotlib import gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from common import *

output_dir = os.path.join(OUTPUT_DIR, 'granular_ascadv1_qualitative')
os.makedirs(output_dir, exist_ok=True)

for dataset_name in ['ascadv1-fixed', 'ascadv1-variable', 'dpav4', 'aes-hd', 'otiait', 'otp']:
    results_dir = os.path.join(OUTPUT_DIR, dataset_name.replace('-', '_'))
    osnr_measurements = dict()
    targets = ['r_out', 'r', 'r_in', 'subbytes__r_out', 'subbytes__r', 'plaintext__key__r_in'] if 'ascadv1' in dataset_name else ['label']
    for target in targets:
        osnr_path = os.path.join(results_dir, 'first_order_parametric_statistical_assessment', f'attack_snr_{target}.npy')
        osnr_measurement = np.load(osnr_path)
        osnr_measurements[target] = osnr_measurement
    all_measurements = np.stack([
        np.load(os.path.join(results_dir, 'all_runs', 'fair', f'seed={seed}', 'all_training', 'leakage_assessment.npy')) for seed in [50, 51, 52, 53, 54]
    ]).mean(axis=0)
    gradvis_measurements = np.stack([
        np.load(os.path.join(results_dir, 'supervised_models_for_attribution', 'classification', f'seed={seed}', 'gradvis.npz'))['attribution'] for seed in [55, 56, 57, 58, 59]
    ]).mean(axis=0)
    occl_measurements = np.stack([
        np.load(os.path.join(results_dir, 'supervised_models_for_attribution', 'classification', f'seed={seed}', '1-occlusion.npz'))['attribution'] for seed in [55, 56, 57, 58, 59]
    ]).mean(axis=0)
    methods = {'ALL (ours)': all_measurements, 'GradVis': gradvis_measurements, '1-occlusion': occl_measurements}
    fig, axes = plt.subplots(3, 6, figsize=(4*6, 4*3))
    for axes_r, (method_name, method_measurements) in zip(axes, methods.items()):
        for ax, target in zip(axes_r, targets):
            x = osnr_measurements[target]
            y = method_measurements

            ax.set_xlabel(r'Oracle leakiness of $X_t$')
            ax.set_ylabel(f'Estimated leakiness by {method_name}')
            ax.plot(x, y, color='blue', marker='.', markersize=5, linestyle='none')
            ax.set_title(f'Target: {target}')
            ax.set_xscale('log')
            ax.set_yscale('log')

            # Add a vertical boxplot on the left
            box_ax = inset_axes(ax, width="10%", height="100%", loc='center left', 
                                bbox_to_anchor=(-0.15, 0, 1, 1), 
                                bbox_transform=ax.transAxes, borderpad=0)
            box_ax.boxplot(y, vert=True, patch_artist=True, widths=0.3)
            box_ax.set_yscale('log')
            box_ax.set_xticks([])  # Remove x-ticks from the boxplot
            box_ax.set_ylim(ax.get_ylim())  # Match scatter plot y-limits
            box_ax.spines['top'].set_visible(False)
            box_ax.spines['right'].set_visible(False)
            box_ax.spines['bottom'].set_visible(False)
            box_ax.spines['left'].set_visible(False)
            box_ax.tick_params(left=False, labelleft=False)

            #ax.set_xlabel(r'Oracle leakiness of $X_t$')
            #ax.set_ylabel(f'Estimated leakiness by {method_name}')
            #ax.plot(osnr_measurements[target], method_measurements, color='blue', marker='.', markersize=1, linestyle='none')
            #ax.set_title(f'Target: {target}')
            #ax.set_xscale('log')
            #ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'scatterplots_{dataset_name}.png'))