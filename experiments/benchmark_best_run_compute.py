"""
Report compute metrics (params, TFLOPs/step, VRAM, wall-clock time/step) for the
best attacker and best localizer of each dataset, using the exact model architecture
and input dimensions of the real runs.

Usage:
    python experiments/benchmark_best_run_compute.py
"""

import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.profiler import ProfilerActivity, profile

from leakage_localization.models.model import Model
from init_things import *

sys.path.insert(0, str(Path(__file__).parent))
from analysis_for_paper import get_best_runs

BATCH_SIZE = 256
N_WARMUP   = 3
N_ITERS    = 10


def load_model_kwargs(trial_path: Path) -> dict:
    with open(trial_path / 'hparams.yaml') as f:
        raw = re.sub(r'!!python/\S+', '', f.read())
    hparams = yaml.safe_load(raw)
    return hparams['model_kwargs']


def build_model(mk: dict) -> Model:
    model = Model(
        input_length=mk['input_length'],
        output_dim=mk.get('output_dim', 256),
        output_count=mk['output_count'],
        grey_box_head=mk.get('grey_box_head'),
        trunk=mk['trunk'],
        position_embedding=mk['position_embedding'],
        pooling=mk['pooling'],
        head=mk['head'],
        fnn_style=mk['fnn_style'],
        patch_size=mk['patch_size'],
        use_fourier_embed=mk.get('use_fourier_embed', False),
        fourier_embed_num_bands=mk.get('fourier_embed_num_bands'),
        fourier_embed_sigma=mk.get('fourier_embed_sigma'),
        embedding_dim=mk['embedding_dim'],
        expansion_factor=mk.get('expansion_factor', 4),
        trunk_blocks=mk['trunk_blocks'],
        register_tokens=mk.get('register_tokens', 0),
        perceiver_latent_dim=mk.get('perceiver_latent_dim'),
        perceiver_self_attn_per_cross_attn_blocks=mk.get('perceiver_self_attn_per_cross_attn_blocks'),
        perceiver_cross_attn_head_count=mk.get('perceiver_cross_attn_head_count'),
        head_count=mk.get('head_count'),
        input_dropout_rate=0.0,
        input_droppatch_rate=0.0,
        hidden_dropout_rate=0.0,
        use_bias=mk.get('use_bias', False),
    )
    return model.cuda().train()


def _forward_backward(model: Model, x: torch.Tensor) -> None:
    out = model(x)
    loss = out.sum()
    loss.backward()
    model.zero_grad(set_to_none=True)


def measure_flops(model: Model, x: torch.Tensor) -> int:
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True) as prof:
        _forward_backward(model, x)
    torch.cuda.synchronize()
    return sum(e.flops for e in prof.key_averages())


def measure_vram_mb(model: Model, x: torch.Tensor) -> float:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _forward_backward(model, x)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2


def measure_wall_time_ms(model: Model, x: torch.Tensor) -> float:
    for _ in range(N_WARMUP):
        _forward_backward(model, x)
        torch.cuda.synchronize()
    times = []
    for _ in range(N_ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _forward_backward(model, x)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return float(np.median(times) * 1000)


def benchmark_run(label: str, trial_path: Path) -> None:
    print(f'\n  [{label}]  {trial_path.name}')
    mk = load_model_kwargs(trial_path)
    print(f'    input_length={mk["input_length"]}  patch_size={mk["patch_size"]}  '
          f'trunk_blocks={mk["trunk_blocks"]}  embedding_dim={mk["embedding_dim"]}')
    torch.manual_seed(0)
    model = build_model(mk)
    x = torch.randn(BATCH_SIZE, 1, mk['input_length'], device='cuda')

    params   = sum(p.numel() for p in model.parameters())
    flops    = measure_flops(model, x)
    wall_ms  = measure_wall_time_ms(model, x)
    vram_mb  = measure_vram_mb(model, x)

    del model, x
    torch.cuda.empty_cache()

    print(f'    params     = {params:,}  ({params/1e6:.2f} M)')
    print(f'    TFLOPs/step = {flops/1e12:.3f}')
    print(f'    VRAM       = {vram_mb/1024:.2f} GB')
    print(f'    time/step  = {wall_ms:.1f} ms')


def main():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    for dataset_id in ['ascadv1-fixed', 'ascadv1-variable', 'ches-ctf-2018']:
        print(f'\n=== {dataset_id} ===')
        best_attack_rv, best_loc_rv = get_best_runs(dataset_id)
        benchmark_run('best attacker', Path(best_attack_rv['path']))
        benchmark_run('best localizer', Path(best_loc_rv['path']))


if __name__ == '__main__':
    main()