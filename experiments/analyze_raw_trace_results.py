import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle

from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from common import *
from utils.baseline_assessments import FirstOrderStatistics
from training_modules.adversarial_leakage_localization import ALLModule

TRIAL_COUNT = 8
fig, axes = plt.subplots(TRIAL_COUNT, 3, figsize=(3*4, TRIAL_COUNT*4))
for trial_idx in range(TRIAL_COUNT):
    checkpoint_path = f'/home/jgammell/Desktop/learning_to_localize_leakage/gautschi_outputs/ascadv1f_raw_trials/all_train/trial_idx={trial_idx}/lightning_logs/version_0/checkpoints/epoch=113-step=20000.ckpt'
    output_path = os.path.join(OUTPUT_DIR, r'raw_trace_visualization')
    os.makedirs(output_path, exist_ok=True)

    all_module = ALLModule.load_from_checkpoint(checkpoint_path, strict=False, map_location='cpu')
    log_gamma = all_module.selection_mechanism.get_log_gamma().detach().numpy().squeeze()
    with open(r'/home/jgammell/Desktop/learning_to_localize_leakage/gautschi_outputs/ascadv1f_raw_trials/snr.pickle', 'rb') as f:
        snr_vals = pickle.load(f)
    snr = np.stack(list(snr_vals.values())).mean(axis=0)
    print(f'Oracle agreement: {spearmanr(log_gamma, snr).statistic}')
    ax = axes[trial_idx, 0]
    ax.plot(snr, color='blue', linestyle=':', marker='.', markersize=1, linewidth=0.2)
    ax.set_xlabel(r'Time $t$')
    ax.set_ylabel(r'SNR: Estimated leakiness of $X_t$')
    ax.set_yscale('log')
    ax = axes[trial_idx, 1]
    ax.plot(np.exp(log_gamma), color='blue', linestyle=':', marker='.', markersize=1, linewidth=0.2)
    ax.set_xlabel(r'Time $t$')
    ax.set_ylabel(r'ALL: Estimated leakiness of $X_t$')
    ax = axes[trial_idx, 2]
    ax.plot(snr, np.exp(log_gamma), color='blue', linestyle='none', marker='.', markersize=1)
    ax.set_xscale('log')
    ax.set_xlabel(r'Ground truth leakiness of $X_t$')
    ax.set_ylabel(r'Estimated leakiness of $X_t$')
fig.tight_layout()
fig.savefig(os.path.join(output_path, f'qualitative_agreement.png'))
plt.close(fig)