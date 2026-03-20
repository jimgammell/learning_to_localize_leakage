from typing import List, Optional
from math import floor, log10

import numpy as np

class PerformanceComparisonTable:
    def __init__(self, random_methods: List[str], parametric_methods: List[str], deep_methods: List[str], our_methods: List[str]):
        self.datasets = ['ASCADv1 (fixed)', 'ASCADv1 (random)', 'DPAv4 (Zaid version)', 'AES-HD', 'OTiAiT', 'OTP']
        self.methods = {
            'random': {method: np.full((len(self.datasets), 2), np.nan, dtype=np.float64) for method in random_methods},
            'parametric': {method: np.full((len(self.datasets), 2), np.nan, dtype=np.float64) for method in parametric_methods},
            'deep': {method: np.full((len(self.datasets), 2), np.nan, dtype=np.float64) for method in deep_methods},
            'ours': {method: np.full((len(self.datasets), 2), np.nan, dtype=np.float64) for method in our_methods}
        }
    
    def record_result(self, method_class: str, method: str, dataset: str, value: float, error: Optional[float] = None):
        assert method_class in self.methods
        assert method in self.methods[method_class]
        assert dataset in self.datasets
        dataset_idx = self.datasets.index(dataset)
        self.methods[method_class][method][dataset_idx, 0] = value
        if error is not None:
            self.methods[method_class][method][dataset_idx, 1] = error

    def result_to_str(self, method_class: str, method: str, dataset: str) -> str:
        assert method_class in self.methods
        assert method in self.methods[method_class]
        assert dataset in self.datasets
        dataset_idx = self.datasets.index(dataset)
        value, error = self.methods[method_class][method][dataset_idx, :]
        if np.isnan(value):
            assert np.isnan(error)
            return ' n/a '
        elif np.isnan(error):
            return f'{value:.3}'
        else:
            assert error > 0
            exponent = floor(log10(error))
            fmt = f'{{:.{-exponent}f}}' if exponent < 0 else '{:.0f}'
            rv = fmt.format(round(value, -exponent))
            return fmt.format(round(value, max(0, -exponent))) + r' \pm ' + fmt.format(round(error, max(0, -exponent)))

    def print_table(self):
        return '\n'.join([
            r'\begin{tabular}{llcccccc}',
            r'\toprule',
            r'& \textbf{Method} & \multicolumn{2}{c}{\textbf{2nd-order datasets}} & \multicolumn{4}{c}{\textbf{1st-order datasets}} \\',
            r'& & ASCADv1 (fixed) & ASCADv1 (random) & DPAv4 (Zaid version) & AES-HD & OTiAiT & OTP \\',
            r'\cmidrule(lr){3-4} \cmidrule{lr}{5-8}',
            *['&'.join(['', f'{method}', *[self.result_to_str('random', method, dataset) for dataset in self.datasets]]) + r'\\' for method in self.methods['random'].keys()],
            r'\midrule',
            r'\multirow{' + f'{len(self.methods['parametric'])}' + r'{2cm}{First-order parametric methods}',
            *['&'.join(['', f'{method}', *[self.result_to_str('parametric', method, dataset) for dataset in self.datasets]]) + r'\\' for method in self.methods['parametric'].keys()],
            r'\midrule',
            r'\multirow{' + f'{len(self.methods['deep'])}' + r'{2cm}{Prior deep learning approaches}'
            *['&'.join(['', f'{method}', *[self.result_to_str('deep', method, dataset) for dataset in self.datasets]])  + r'\\' for method in self.methods['deep'].keys()],
            r'\midrule',
            *['&'.join(['', f'{method}', *[self.result_to_str('ours', method, dataset) for dataset in self.datasets]]) + r'\\' for method in self.methods['ours'].keys()],
            r'\bottomrule',
            r'\end{tabular}'
        ])