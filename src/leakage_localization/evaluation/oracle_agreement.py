from pathlib import Path
from typing import Dict, get_args, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.stats import spearmanr, f as f_dist
from sklearn.metrics import roc_auc_score

from leakage_localization.datasets.common import DATASET, PARTITION

class OracleAgreement:
    def __init__(
            self,
            snr_dir: Path,
            dataset: DATASET,
    ):
        self.snr_dir = snr_dir
        self.dataset = dataset

        assert isinstance(self.snr_dir, Path) and self.snr_dir.exists()
        assert self.dataset in get_args(DATASET)

        _masked_vars = [
            'r', 'r_in', 'r_out',
            'subbytes__xor__r', 'subbytes__xor__r_out',
            'p__xor__k__xor__r_in', 'p__xor__k__xor__r',
        ]
        if self.dataset == 'ascadv1-fixed':
            self.byte_count = 16
            self.feature_count = 100_000
            self.num_classes = 256
            self.n_traces = {'attack': 10_000, 'profile': 50_000}
            self.variables = {
                **{idx: ['subbytes'] for idx in range(2)},
                **{idx: _masked_vars for idx in range(2, 16)},
            }
        elif self.dataset == 'ascadv1-variable':
            self.byte_count = 16
            self.feature_count = 250_000
            self.num_classes = 256
            self.n_traces = {'attack': 100_000, 'profile': 200_000}
            self.variables = {
                **{idx: ['subbytes'] for idx in range(2)},
                **{idx: _masked_vars for idx in range(2, 16)},
            }
        else:
            raise NotImplementedError(f'No implementation for key {dataset}')
        self.oracle_leakiness = self.get_oracle_leakiness('attack')
    
    def get_oracle_leakiness(self, partition: PARTITION) -> NDArray[np.float32]:
        oracle_leakiness = np.zeros((self.byte_count, self.feature_count), dtype=np.float32)
        for byte_idx, _variables in self.variables.items():
            for variable in _variables:
                snr_path = self.snr_dir / f'{variable}.{partition}.npy'
                assert snr_path.exists()
                snr = np.load(snr_path)
                snr_byte_count, snr_feature_count = snr.shape
                assert self.feature_count == snr_feature_count
                assert snr_byte_count in {1, self.byte_count}
                oracle_leakiness[byte_idx, :] += snr[min(byte_idx, snr_byte_count - 1), :]
        return oracle_leakiness
    
    def get_threshold(
            self,
            partition: PARTITION,
            percentile: float = 0.9999,
            snr_threshold: Optional[float] = None,
    ) -> float:
        """Return the SNR threshold, either from a direct value or derived from
        the F-distribution null at the given percentile."""
        if snr_threshold is not None:
            return snr_threshold
        n = self.n_traces[partition]
        df1 = self.num_classes - 1
        df2 = n - self.num_classes
        return float(f_dist.ppf(percentile, df1, df2) * df1 / df2)

    def get_binary_labels(
            self,
            partition: PARTITION,
            percentile: float = 0.9999,
            snr_threshold: Optional[float] = None,
    ) -> NDArray[np.bool_]:
        """Binary leakage labels via per-variable F-distribution threshold.

        Under H₀ (no leakage), SNR x df₂/df₁ ~ F(df₁, df₂) where
        df₁ = num_classes-1 and df₂ = N-num_classes.

        A timestep is labelled leaky for byte b if ANY variable relevant to
        byte b has SNR above the chosen percentile of this null distribution.
        Note: shared single-byte variables (e.g. r_in, r_out) will contribute
        the same leaky timesteps to all bytes that use them.

        Pass snr_threshold to override the percentile-derived threshold with a
        fixed SNR cutoff, which is useful when the F-null assumption is violated
        for some variables (e.g. elevated SNR floors).
        """
        threshold = self.get_threshold(partition, percentile, snr_threshold)
        labels = np.zeros((self.byte_count, self.feature_count), dtype=bool)
        for byte_idx, var_names in self.variables.items():
            for var_name in var_names:
                snr_path = self.snr_dir / f'{var_name}.{partition}.npy'
                assert snr_path.exists(), f'SNR file not found: {snr_path}'
                snr = np.load(snr_path)
                row = min(byte_idx, snr.shape[0] - 1)
                labels[byte_idx] |= (snr[row] > threshold)
        return labels

    def get_auroc(
            self,
            x: NDArray[np.floating],
            partition: PARTITION = 'attack',
            percentile: float = 0.9999,
            snr_threshold: Optional[float] = None,
    ) -> NDArray[np.floating]:
        """Per-byte AUROC of x against binary leakage labels."""
        byte_count, feature_count = x.shape
        assert byte_count == self.byte_count
        assert feature_count == self.feature_count
        labels = self.get_binary_labels(partition, percentile, snr_threshold)
        auroc = np.full(byte_count, np.nan, dtype=np.float64)
        for b in range(byte_count):
            pos = labels[b].sum()
            if 1 < pos < feature_count - 1:
                auroc[b] = roc_auc_score(labels[b], x[b])
        return auroc

    def get_full_spearman(self, x: NDArray[np.floating]) -> float:
        """Spearman correlation of the byte-averaged attribution against the
        byte-averaged oracle leakiness.  This cannot be derived from per-byte
        Spearman values because correlation does not commute with averaging."""
        byte_count, feature_count = x.shape
        assert byte_count == self.byte_count
        assert feature_count == self.feature_count
        return float(spearmanr(x.mean(axis=0), self.oracle_leakiness.mean(axis=0)).statistic)

    def get_full_auroc(
            self,
            x: NDArray[np.floating],
            partition: PARTITION = 'attack',
            percentile: float = 0.9999,
            snr_threshold: Optional[float] = None,
    ) -> float:
        """AUROC of the byte-averaged attribution against union-of-bytes binary
        leakage labels.  A timestep is considered leaky if it is leaky for any
        byte; the score is the mean attribution across bytes."""
        byte_count, feature_count = x.shape
        assert byte_count == self.byte_count
        assert feature_count == self.feature_count
        labels = self.get_binary_labels(partition, percentile, snr_threshold)  # (byte_count, feature_count)
        union_labels = labels.any(axis=0)                        # (feature_count,)
        x_mean = x.mean(axis=0)                                  # (feature_count,)
        pos = int(union_labels.sum())
        if 1 < pos < feature_count - 1:
            return float(roc_auc_score(union_labels, x_mean))
        return float('nan')

    def __call__(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        byte_count, feature_count = x.shape
        assert byte_count == self.byte_count
        assert feature_count == self.feature_count
        oracle_agreement = np.array([
            spearmanr(x[byte_idx, :], self.oracle_leakiness[byte_idx, :]).statistic for byte_idx in range(byte_count)
        ])
        return oracle_agreement

    def get_random_oracle_agreement(self) -> NDArray[np.floating]:
        random_leakiness = np.random.rand(*self.oracle_leakiness.shape)
        return self(random_leakiness)

    def get_profiling_oracle_agreement(self) -> NDArray[np.floating]:
        profiling_oracle_leakiness = self.get_oracle_leakiness('profile')
        return self(profiling_oracle_leakiness)