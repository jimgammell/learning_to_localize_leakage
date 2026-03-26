"""Temporary: profile template attack and verify correctness on all bytes."""
import time
import numpy as np
from init_things import *
from utils.load_data import load_numpy_dataset
from leakage_localization.evaluation.ta_mtd import compute_ta_mtd

dataset_id = 'ascadv1-fixed'
estimates = np.load('outputs/ascadv1_fixed/reg_sweep/mixup_0.2/seed_0/gradvis.npy')
print(f'estimates shape: {estimates.shape}')

t0 = time.time()
profiling_set = load_numpy_dataset(dataset_id, 'profile')
attack_set = load_numpy_dataset(dataset_id, 'attack')
print(f'Load datasets: {time.time()-t0:.1f}s')

t0 = time.time()
ta_mtd = compute_ta_mtd(estimates, profiling_set, attack_set)
total = time.time() - t0
print(f'\nTotal compute_ta_mtd: {total:.1f}s ({total/16:.1f}s per byte)')
print(f'ta_mtd shape: {ta_mtd.shape}')
print(f'ta_mtd per byte: {ta_mtd}')
print(f'Mean ta_mtd: {ta_mtd.mean():.1f}')
print(f'All bytes succeeded (MTD < 10001): {(ta_mtd < 10001).all()}')