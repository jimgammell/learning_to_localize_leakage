from typing import Dict, Any, Optional, Literal
import os
import pickle
import json

import numpy as np
from torch.utils.data import Dataset, DataLoader

from common import *
from training_modules.supervised_deep_sca import SupervisedTrainer, SupervisedModule
from utils.aes_multi_trace_eval import AESMultiTraceEvaluator
from utils.baseline_assessments.neural_net_attribution import NeuralNetAttribution
from utils.baseline_assessments.occpoi import OccPOI

def load_trained_supervised_model(
    model_dir: str, as_lightning_module: bool = False
):
    checkpoint_path = os.path.join(model_dir, 'early_stop_checkpoint.ckpt')
    lightning_module = SupervisedModule.load_from_checkpoint(checkpoint_path, map_location='cpu')
    if as_lightning_module:
        return lightning_module
    else:
        return lightning_module.classifier

def train_supervised_model(
    output_dir: str,
    profiling_dataset: Dataset,
    attack_dataset: Dataset,
    training_kwargs: Optional[Dict[str, Any]] = None,
    max_steps: int = 1000,
    seed: int = 0,
    reference_leakage_assessment: Optional[np.ndarray] = None
):
    if training_kwargs is None:
        training_kwargs = {}
    trainer = SupervisedTrainer(profiling_dataset, attack_dataset, default_training_module_kwargs=training_kwargs, reference_leakage_assessment=reference_leakage_assessment)
    set_seed(seed)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    trainer.run(output_dir, max_steps=max_steps, plot_metrics_over_time=reference_leakage_assessment is not None)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    np.save(os.path.join(output_dir, 'training_time.npy'), elapsed_time)

# Train 50 supervised models on the dataset of interest
def run_supervised_hparam_sweep(
    output_dir: str, # where things will be saved
    profiling_dataset: Dataset,
    attack_dataset: Dataset,
    training_kwargs: Optional[Dict[str, Any]] = None, # training kwargs to use unless otherwise specified by hyperparameter config.
    trial_count: int = 50, # number of hparam configurations to try
    max_steps: int = 1000, # number of training steps per trial
    starting_seed: int = 0 # each trial will use the seed trial_idx+starting_seed
):
    if training_kwargs is None:
        training_kwargs = {}
    trainer = SupervisedTrainer(profiling_dataset, attack_dataset, default_training_module_kwargs=training_kwargs)
    trainer.hparam_tune(output_dir, trial_count=trial_count, max_steps=max_steps, starting_seed=starting_seed)

def get_best_supervised_model_hparams(sweep_dir: str):
    best_model_dir = None
    best_val_rank, best_val_loss = float('inf'), float('inf')
    for model_dir in os.listdir(sweep_dir):
        assert 'training_curves.pickle' in os.listdir(os.path.join(sweep_dir, model_dir))
        assert 'early_stop_checkpoint.ckpt' in os.listdir(os.path.join(sweep_dir, model_dir))
        with open(os.path.join(sweep_dir, model_dir, 'training_curves.pickle'), 'rb') as f:
            training_curves = pickle.load(f)
        val_rank_over_time = training_curves['val_rank'][-1]
        val_loss_over_time = training_curves['val_loss'][-1]
        early_stop_idx = np.argmin(val_rank)
        val_rank = val_rank_over_time[early_stop_idx]
        val_loss = val_loss_over_time[early_stop_idx]
        if (val_rank < best_val_rank) or (val_rank == best_val_rank and val_loss < best_val_loss):
            best_model_dir = model_dir
            best_val_rank = val_rank
            best_val_loss = val_loss
    with open(os.path.join(best_model_dir, 'hparams.json'), 'r') as f:
        best_hparams = json.load(f)
    return best_hparams

# Create plots showing the performance of a trained supervised model
def eval_on_attack_dataset(model_dir: str, attack_dataset: Dataset, dataset_name: str):
    if os.path.exists(os.path.join(model_dir, 'attack_performance.npy')):
        rv = np.load(os.path.join(model_dir, 'attack_performance.npy'))
    else:
        dataloader = DataLoader(attack_dataset, batch_size=len(attack_dataset), num_workers=4, dataset_named=dataset_name)
        model = load_trained_supervised_model(model_dir, as_lightning_module=False)
        evaluator = AESMultiTraceEvaluator(dataloader, model, seed=0, dataset_name=dataset_name)
        rv = evaluator()
        np.save(os.path.join(model_dir, 'attack_performance.npy'), rv)
    return rv

# Compute various neural net attribution leakage assessments given a trained model directory
def attribute_neural_net(
    model_dir, profiling_dataset: Dataset, attack_dataset: Dataset, dataset_name: str,
    compute_gradvis: bool = False, compute_saliency: bool = False, compute_inputxgrad: bool = False,
    compute_lrp: bool = False, compute_occlusion: List[int] = [], compute_second_order_occlusion: List[int] = [],
    compute_occpoi: bool = False, compute_extended_occpoi: bool = False
):
    profiling_dataloader = DataLoader(profiling_dataset, batch_size=len(profiling_dataset), num_workers=4)
    attack_dataloader = DataLoader(attack_dataset, batch_size=len(attack_dataset), num_workers=4)
    model = load_trained_supervised_model(model_dir, as_lightning_module=False)
    neural_net_attributor = NeuralNetAttribution(profiling_dataloader, model, seed=0, device='cuda' if torch.cuda.is_available() else 'cpu')
    occpoi_computor = OccPOI(attack_dataloader, model, seed=0, device='cuda' if torch.cuda.is_available() else 'cpu', dataset_name=dataset_name)
    def compute_attribution(attribution_fn: Callable, filename: str):
        if not os.path.exists(os.path.join(model_dir, filename)):
            set_seed(0)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            attribution = attribution_fn()
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
            rv = {'attribution': attribution, 'elapsed_time': elapsed_time}
            np.savez(os.path.join(model_dir, filename), **rv)
    if compute_gradvis:
        compute_attribution(lambda: neural_net_attributor.compute_gradvis(), 'gradvis.npz')
    if compute_saliency:
        compute_attribution(lambda: neural_net_attributor.compute_saliency(), 'saliency.npz')
    if compute_inputxgrad:
        compute_attribution(lambda: neural_net_attributor.compute_inputxgrad(), 'occpoi.npz')
    if compute_lrp:
        compute_attribution(lambda: neural_net_attributor.compute_lrp(), 'lrp.npz')
    for window_size in compute_occlusion:
        compute_attribution(lambda: neural_net_attributor.compute_n_occlusion(window_size), f'{window_size}-occlusion.npz')
    for window_size in compute_second_order_occlusion:
        compute_attribution(lambda: neural_net_attributor.compute_second_order_occlusion(window_size=window_size), f'{window_size}-second-order-occlusion.npz')
    if compute_occpoi:
        compute_attribution(lambda: occpoi_computor(extended=False), 'occpoi.npz')
    if compute_extended_occpoi:
        compute_attribution(lambda: occpoi_computor(extended=True), 'extended-occpoi.npz')