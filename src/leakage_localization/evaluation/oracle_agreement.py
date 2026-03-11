from pathlib import Path
from typing import Dict, List

import numpy as np
from numpy.typing import NDArray
from scipy.stats import spearmanr

from leakage_localization.datasets.common import PARTITION

class _ComputeOracleAgreement:
    def __init__(
            self,
            snr_dir: Path,
            partition: PARTITION,
            variables: Dict[int, List[str]]
    ):
        self.snr_dir = snr_dir
        self.partition = partition
        self.variables = variables
    
    def __call__(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        byte_count, feature_count = x.shape
        oracle_assessment = np.zeros((byte_count, feature_count), dtype=x.dtype)
        for byte_idx, _variables in self.variables.items():
            for variable in _variables:
                snr_path = self.snr_dir / f'{variable}.{self.partition}.npy'
                assert snr_path.exists()
                snr = np.load(snr_path)
                snr_byte_count, snr_feature_count = snr.shape
                assert feature_count == snr_feature_count
                assert snr_byte_count in {1, byte_count}
                oracle_assessment[byte_idx, :] += snr[min(byte_idx, snr_byte_count - 1), :]
        oracle_agreement = np.array([
            spearmanr(x[byte_idx, :], oracle_assessment[byte_idx, :]).statistic for byte_idx in range(byte_count)
        ])
        return oracle_agreement

class ComputeASCADv1OracleAgreement(_ComputeOracleAgreement):
    def __init__(
        self,
        snr_dir: Path,
        partition: PARTITION
    ):
        super().__init__(
            snr_dir,
            partition,
            variables={
                **{idx: ['subbytes'] for idx in range(2)},
                **{idx: [
                    'p__xor__k__xor__r_in', 'r_in',
                    'subbytes__xor__r', 'r',
                    'subbytes__xor__r_out', 'r_out'
                    ] for idx in range(2, 16)
                }
            }
        )