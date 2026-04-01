"""Compare RISE convergence vs. GradVis on a trained model.

Runs RISE for a configurable number of passes, evaluating AUROC after each
checkpoint interval. GradVis is computed once as a reference baseline.

Usage:
    python experiments/compare_rise_vs_gradvis.py \
        --ckpt-path ./outputs/ascadv1_fixed/seed_0/best_val_rank.ckpt \
        --max-passes 50 \
        --checkpoint-every 5 \
        --dest ./outputs/rise_convergence
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from init_things import *
from utils.load_things import load_numpy_dataset, load_torch_dataset, construct_loaders, load_trained_model
from leakage_localization.deep_attribution.attributor import Attributor
from leakage_localization.deep_attribution.rise_attributor import RISEAttributor
from leakage_localization.evaluation import OracleAgreement


def compute_auroc(attr_np, oracle):
    """Returns mean per-byte AUROC, ignoring NaN bytes."""
    auroc = oracle.get_auroc(attr_np, partition="attack")
    return float(np.nanmean(auroc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=Path, required=True)
    parser.add_argument("--max-passes", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=5,
                        help="Evaluate AUROC every this many passes")
    parser.add_argument("--mask-prob", type=float, default=0.5)
    parser.add_argument("--dest", type=Path, default=None)
    args = parser.parse_args()

    ckpt_path: Path = args.ckpt_path
    dest: Path = args.dest or ckpt_path.parent / "rise_convergence"
    dest.mkdir(parents=True, exist_ok=True)

    # Infer dataset from config.yaml in checkpoint directory
    config_path = ckpt_path.parent / "config.yaml"
    assert config_path.exists(), f"Expected config.yaml at {config_path}"
    with open(config_path) as f:
        config_kw = safe_load_yaml(f)
    from utils.training_config import SupervisedTrainingConfig
    config = SupervisedTrainingConfig(**config_kw)
    dataset_id = config.data.id

    dataset_kwargs = {
        "target_byte": config.data.target_byte,
        "target_variable": config.data.target_variable,
    }
    profiling_set = load_numpy_dataset(dataset_id, "profile", **dataset_kwargs)
    profile_torch = load_torch_dataset(dataset_id, "profile", **dataset_kwargs)
    profile_loader, = construct_loaders([], [profile_torch])

    module = load_trained_model(ckpt_path, profiling_set)
    module.to("cuda")
    module.eval()

    snr_dir = get_output_dir(dataset_id) / "snr"
    oracle = OracleAgreement(snr_dir, dataset_id)

    # ---- GradVis reference (one pass) ----
    logging.info("Computing GradVis attribution...")
    attributor = Attributor(module)
    gradvis_attr = attributor("gradvis", profile_loader, show_progress_bar=True)
    gradvis_np = gradvis_attr.cpu().numpy().astype(np.float32)
    np.save(dest / "gradvis.npy", gradvis_np)
    gradvis_auroc = compute_auroc(gradvis_np, oracle)
    logging.info(f"GradVis mean AUROC: {gradvis_auroc:.4f}")

    # ---- RISE convergence ----
    rise = RISEAttributor(module, profile_loader, mask_prob=args.mask_prob)

    pass_checkpoints = []
    auroc_checkpoints = []

    total_passes = 0
    while total_passes < args.max_passes:
        passes_this_round = min(args.checkpoint_every, args.max_passes - total_passes)
        rise.run_passes(passes_this_round, show_progress=True)
        total_passes = rise.n_passes

        attr_np = rise.attribution.numpy().astype(np.float32)
        auroc = compute_auroc(attr_np, oracle)
        pass_checkpoints.append(total_passes)
        auroc_checkpoints.append(auroc)
        logging.info(f"RISE passes={total_passes}  mean AUROC={auroc:.4f}")

        # Save latest attribution
        np.save(dest / f"rise_passes{total_passes:04d}.npy", attr_np)

    # Save convergence data
    np.savez(
        dest / "rise_convergence.npz",
        passes=np.array(pass_checkpoints),
        auroc=np.array(auroc_checkpoints),
        gradvis_auroc=np.array(gradvis_auroc),
    )

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(pass_checkpoints, auroc_checkpoints, marker="o", label="RISE")
    ax.axhline(gradvis_auroc, color="darkorange", linestyle="--", label="GradVis (1 pass)")
    ax.set_xlabel("RISE passes over profiling set")
    ax.set_ylabel("Mean per-byte AUROC")
    ax.set_title(f"RISE convergence vs GradVis ({dataset_id})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(dest / "rise_convergence.pdf")
    plt.close(fig)
    logging.info(f"Saved results to {dest}")


if __name__ == "__main__":
    main()
