from typing import Optional, Literal, List
from pathlib import Path

from torch.utils.data import DataLoader
import lightning
from lightning import Trainer, LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

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
    callbacks = []
    if early_stop_metric is not None:
        best_ckpt_name = f'best_{early_stop_metric}'.replace('/', '_')
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
    else:
        best_ckpt_name = 'latest'
    final_checkpoint_callback = ModelCheckpoint(
        dirpath=dest,
        filename='latest',
        enable_version_counter=False
    )
    callbacks.append(final_checkpoint_callback)
    if aux_callbacks is not None:
        callbacks += aux_callbacks
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
    trainer.fit(training_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    try:
        trainer.test(training_module, dataloaders=test_loader, ckpt_path='best', weights_only=False)
    except Exception as e:
        print(f"Warning: test step failed with {type(e).__name__}: {e}")