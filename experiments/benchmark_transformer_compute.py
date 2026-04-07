"""
Benchmark transformer compute cost vs. key hyperparameters.

Sweeps patch_count (with fixed input_length=2^18, so patch_size varies inversely),
layer_count, and embedding_dim, measuring per-forward+backward-pass FLOPs, wall-clock
time, parameter count, and peak VRAM.

With fixed input_length, increasing patch_count shrinks patch_size: the transformer
sees a longer sequence (more FLOPs) but the patchifier projection has fewer parameters.

Usage:
    python experiments/benchmark_transformer_compute.py [--output-dir PATH]
"""

from typing import Optional
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

from leakage_localization.models.model import Model
from init_things import *

# ── Fixed benchmark parameters ────────────────────────────────────────────────
BATCH_SIZE       = 256
FIXED_INPUT_LEN  = 2**18   # 262 144; patch_size = FIXED_INPUT_LEN // patch_count
OUTPUT_COUNT     = 16      # number of output variables predicted (as in practice)
OUTPUT_DIM       = 256
EXPANSION_FACTOR = 4
N_WARMUP         = 3
N_ITERS          = 10

# ── Base config ───────────────────────────────────────────────────────────────
BASE_PATCH_COUNT   = 64
BASE_LAYER_COUNT   = 8
BASE_EMBEDDING_DIM = 512

# ── Sweeps (factors of 2 around base) ────────────────────────────────────────
PATCH_COUNTS   = [8, 16, 32, 64, 128, 256]
LAYER_COUNTS   = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
EMBEDDING_DIMS = [128, 256, 384, 512, 640, 768, 896, 1024, 1280, 1536]


# ── Model helpers ─────────────────────────────────────────────────────────────

def build_model(patch_count: int, layer_count: int, embedding_dim: int) -> Model:
    patch_size = FIXED_INPUT_LEN // patch_count
    assert patch_size % 2 == 0, f"patch_size={patch_size} must be even (got patch_count={patch_count})"
    model = Model(
        input_length=FIXED_INPUT_LEN,
        output_dim=OUTPUT_DIM,
        output_count=OUTPUT_COUNT,
        grey_box_head=None,
        trunk='transformer',
        position_embedding='rope',
        pooling='token',
        head='tied',
        fnn_style='mlp',
        patch_size=patch_size,
        use_fourier_embed=False,
        fourier_embed_num_bands=None,
        fourier_embed_sigma=None,
        embedding_dim=embedding_dim,
        expansion_factor=EXPANSION_FACTOR,
        trunk_blocks=layer_count,
        register_tokens=0,
        perceiver_latent_dim=None,
        perceiver_self_attn_per_cross_attn_blocks=None,
        perceiver_cross_attn_head_count=None,
        head_count=None,  # defaults to embedding_dim // 64
        input_dropout_rate=0.0,
        input_droppatch_rate=0.0,
        hidden_dropout_rate=0.0,
        use_bias=False,
    )
    return model.cuda().train()


def make_input(patch_count: int) -> torch.Tensor:
    return torch.randn(BATCH_SIZE, 1, FIXED_INPUT_LEN, device='cuda')


def count_params(model: Model) -> int:
    return sum(p.numel() for p in model.parameters())


def _forward_backward(model: Model, x: torch.Tensor) -> None:
    out = model(x)
    loss = out.sum()
    loss.backward()
    model.zero_grad(set_to_none=True)


# ── Metric measurements ───────────────────────────────────────────────────────

def measure_flops(model: Model, x: torch.Tensor) -> int:
    """
    FLOPs for one forward + backward pass, captured via torch.profiler.

    torch.profiler with_flops=True counts flops for supported operators
    (primarily matrix multiplications and convolutions) in both forward and
    backward kernels. The count may be an underestimate for unsupported ops.
    """
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_flops=True,
    ) as prof:
        _forward_backward(model, x)
    torch.cuda.synchronize()
    return sum(e.flops for e in prof.key_averages())


def measure_vram_mb(model: Model, x: torch.Tensor) -> float:
    """Peak VRAM (MB) over one forward + backward pass."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _forward_backward(model, x)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2


def measure_wall_time_ms(model: Model, x: torch.Tensor) -> float:
    """Median wall-clock time (ms) per forward + backward pass over N_ITERS."""
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


# ── Per-config orchestration ──────────────────────────────────────────────────

def benchmark_config(patch_count: int, layer_count: int, embedding_dim: int) -> dict:
    patch_size = FIXED_INPUT_LEN // patch_count
    base = {
        'patch_count':   patch_count,
        'layer_count':   layer_count,
        'embedding_dim': embedding_dim,
        'patch_size':    patch_size,
    }
    try:
        torch.manual_seed(0)
        model  = build_model(patch_count, layer_count, embedding_dim)
        x      = make_input(patch_count)
        params = count_params(model)
        flops  = measure_flops(model, x)
        wall_ms = measure_wall_time_ms(model, x)
        vram_mb = measure_vram_mb(model, x)

        del model, x
        torch.cuda.empty_cache()

        return {**base,
                'param_count':  params,
                'flops':        flops,
                'wall_time_ms': wall_ms,
                'vram_mb':      vram_mb,
                'oom':          False}

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {**base,
                'param_count':  None,
                'flops':        None,
                'wall_time_ms': float('nan'),
                'vram_mb':      float('nan'),
                'oom':          True}


def _fmt(r: dict) -> str:
    if r['oom']:
        return 'OOM'
    return (
        f"params={r['param_count']:>12,}  "
        f"flops={r['flops']:.3e}  "
        f"vram={r['vram_mb']:>8.1f} MB  "
        f"time={r['wall_time_ms']:>7.1f} ms"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output-dir', type=Path, default=None
    )
    args = parser.parse_args()

    output_dir: Optional[Path] = args.output_dir
    if output_dir is None:
        output_dir = OUTPUTS_ROOT / 'compute_benchmark'
    assert isinstance(output_dir, Path)
    output_dir.mkdir(exist_ok=True, parents=True)

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    results = []

    # ── Sweep: patch_count ────────────────────────────────────────────────────
    print(
        f'=== Sweeping patch_count  '
        f'(layer_count={BASE_LAYER_COUNT}, embedding_dim={BASE_EMBEDDING_DIM}) ==='
    )
    for pc in PATCH_COUNTS:
        ps = FIXED_INPUT_LEN // pc
        print(f'  patch_count={pc:>4}  patch_size={ps:>6} ... ', end='', flush=True)
        r = benchmark_config(pc, BASE_LAYER_COUNT, BASE_EMBEDDING_DIM)
        r['sweep_var'] = 'patch_count'
        results.append(r)
        print(_fmt(r))

    # ── Sweep: layer_count ────────────────────────────────────────────────────
    print(
        f'\n=== Sweeping layer_count  '
        f'(patch_count={BASE_PATCH_COUNT}, embedding_dim={BASE_EMBEDDING_DIM}) ==='
    )
    for lc in LAYER_COUNTS:
        print(f'  layer_count={lc:>3} ... ', end='', flush=True)
        r = benchmark_config(BASE_PATCH_COUNT, lc, BASE_EMBEDDING_DIM)
        r['sweep_var'] = 'layer_count'
        results.append(r)
        print(_fmt(r))

    # ── Sweep: embedding_dim ──────────────────────────────────────────────────
    print(
        f'\n=== Sweeping embedding_dim  '
        f'(patch_count={BASE_PATCH_COUNT}, layer_count={BASE_LAYER_COUNT}) ==='
    )
    for ed in EMBEDDING_DIMS:
        print(f'  embedding_dim={ed:>5} ... ', end='', flush=True)
        r = benchmark_config(BASE_PATCH_COUNT, BASE_LAYER_COUNT, ed)
        r['sweep_var'] = 'embedding_dim'
        results.append(r)
        print(_fmt(r))

    # ── Pack results into arrays and save as .npz ─────────────────────────────
    def _arr(key, dtype=None):
        vals = [r[key] for r in results]
        return np.array(vals, dtype=dtype)

    out_path = output_dir / 'results.npz'
    np.savez(
        out_path,
        sweep_var   = _arr('sweep_var'),
        patch_count = _arr('patch_count',   np.int64),
        layer_count = _arr('layer_count',   np.int64),
        embedding_dim = _arr('embedding_dim', np.int64),
        patch_size  = _arr('patch_size',    np.int64),
        param_count = _arr('param_count',   np.float64),  # float to allow NaN on OOM
        flops       = _arr('flops',         np.float64),
        wall_time_ms = _arr('wall_time_ms', np.float64),
        vram_mb     = _arr('vram_mb',       np.float64),
        oom         = _arr('oom',           bool),
    )
    print(f'\nResults saved to {out_path}')


if __name__ == '__main__':
    main()