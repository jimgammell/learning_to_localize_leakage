from pathlib import Path
from typing import get_args, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.stats import spearmanr

from leakage_localization.datasets import DATASET, PARTITION

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

        if self.dataset == 'ascadv1-fixed':
            self.byte_count = 16
            self.feature_count = 100_000
            self.variables={
                **{idx: ['subbytes'] for idx in range(2)},
                **{idx: [
                    'p__xor__k__xor__r_in', 'r_in',
                    'subbytes__xor__r', 'r',
                    'subbytes__xor__r_out', 'r_out'
                    ] for idx in range(2, 16)
                }
            }
        elif self.dataset == 'ascadv1-variable':
            self.byte_count = 16
            self.feature_count = 250_000
            self.variables={
                **{idx: ['subbytes'] for idx in range(2)},
                **{idx: [
                    'p__xor__k__xor__r_in', 'r_in',
                    'subbytes__xor__r', 'r',
                    'subbytes__xor__r_out', 'r_out'
                    ] for idx in range(2, 16)
                }
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