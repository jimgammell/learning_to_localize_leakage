import shutil

from matplotlib import pyplot as plt
import torch

latex_installed = shutil.which('latex') is not None
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'Times New Roman'
})
if shutil.which('latex') is not None:
    plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{times} \usepackage{amsmath} \usepackage{amssymb}'
    })
else:
    print('Warning: Latex not installed. Plots might look ugly.')

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    gpu_properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    arch = 10*gpu_properties.major + gpu_properties.minor
    if arch >= 70:
        torch.set_float32_matmul_precision('high')