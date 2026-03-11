from typing import Optional, Literal, List
from pathlib import Path
import shutil

import torch
import pandas as pd
from torch.utils.data import DataLoader
import lightning
from lightning import Trainer, LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

def _is_training_complete(ckpt_path: Path, total_steps: int) -> bool:
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    return ckpt.get('global_step', 0) >= total_steps

def train_supervised_model(
        *,
        dest: Path, training_module: LightningModule,
        train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
        total_steps: int,
        grad_clip_val: Optional[float] = None,
        accumulate_grad_batches: int = 1,
        early_stop_metric: Optional[str] = 'val_rank', early_stop_mode: Literal['min', 'max'] = 'max',
        aux_callbacks: Optional[List[lightning.Callback]] = None
):
    latest_ckpt_path = dest / 'latest.ckpt'
    if early_stop_metric is not None:
        best_ckpt_name = f'best_{early_stop_metric}'.replace('/', '_')
    else:
        best_ckpt_name = 'latest'

    # Skip training entirely if already complete
    if latest_ckpt_path.exists() and _is_training_complete(latest_ckpt_path, total_steps):
        print(f'Training already complete (global_step >= {total_steps}). Skipping to test evaluation.')
    else:
        callbacks = []
        if early_stop_metric is not None:
            early_stop_callback = ModelCheckpoint(
                monitor=early_stop_metric,
                mode=early_stop_mode,
                save_top_k=1,
                dirpath=dest,
                filename=best_ckpt_name,
                verbose=True,
                enable_version_counter=False
            )
            callbacks.append(early_stop_callback)
        final_checkpoint_callback = ModelCheckpoint(
            dirpath=dest,
            filename='latest',
            enable_version_counter=False,
            every_n_epochs=1
        )
        callbacks.append(final_checkpoint_callback)
        if aux_callbacks is not None:
            callbacks += aux_callbacks

        # Back up existing metrics before CSVLogger overwrites them on resume
        metrics_csv = dest / 'metrics.csv'
        metrics_backup = dest / 'metrics.backup.csv'
        is_resuming = latest_ckpt_path.exists()
        if is_resuming and metrics_csv.exists():
            shutil.copy2(metrics_csv, metrics_backup)

        logger = CSVLogger(
            save_dir=dest,
            name='',
            version=''
        )
        trainer = Trainer(
            accelerator='gpu',
            precision='bf16-mixed',
            logger=logger,
            callbacks=callbacks,
            max_steps=total_steps,
            accumulate_grad_batches=accumulate_grad_batches,
            gradient_clip_val=grad_clip_val,
            default_root_dir=dest,
        )
        if not is_resuming:
            latest_ckpt_path = None
        trainer.fit(training_module, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=latest_ckpt_path, weights_only=False)

        # Merge old and new metrics after resume
        if metrics_backup.exists():
            old_df = pd.read_csv(metrics_backup)
            new_df = pd.read_csv(metrics_csv)
            merged = pd.concat([old_df, new_df], ignore_index=True)
            merged.to_csv(metrics_csv, index=False)
            metrics_backup.unlink()

    best_ckpt_path = dest / f'{best_ckpt_name}.ckpt'
    test_ckpt_path = best_ckpt_path if best_ckpt_path.exists() else dest / 'latest.ckpt'
    try:
        test_trainer = Trainer(
            accelerator='gpu',
            precision='bf16-mixed',
            logger=False,
            default_root_dir=dest,
        )
        test_trainer.test(training_module, dataloaders=test_loader, ckpt_path=str(test_ckpt_path), weights_only=False)
    except Exception as e:
        print(f"Warning: test step failed with {type(e).__name__}: {e}")