from experiments.initialization import *
from leakage_localization.datasets.ascadv1 import ASCADv1_NumpyDataset

dataset = ASCADv1_NumpyDataset(
    root=ASCADV1_VARIABLE_ROOT,
    partition='profile',
    variable_key=True,
    cropped_traces=False,
    binary_trace_file=True
)
print(dataset)
mean, var = dataset.get_trace_mean_and_variance(use_progress_bar=True)