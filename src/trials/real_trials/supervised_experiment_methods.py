from typing import Dict, Any, Optional, Literal
import os
from collections import defaultdict
import pickle
import json
from tqdm import tqdm

import numpy as np
from scipy.stats import spearmanr
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader

from common import *
from datasets.data_module import DataModule
from training_modules.supervised_deep_sca import SupervisedTrainer, SupervisedModule
from utils.aes_multi_trace_eval import AESMultiTraceEvaluator
from utils.baseline_assessments.neural_net_attribution import NeuralNetAttribution
from utils.baseline_assessments.occpoi import OccPOI
from . import evaluation_methods

def get_dataloader(profiling_dataset: Dataset, attack_dataset: Dataset, split: Literal['profile', 'attack'], **kwargs):
    data_module = DataModule(profiling_dataset, attack_dataset, **kwargs)
    if split == 'profile':
        return data_module.profiling_dataloader()
    elif split == 'attack':
        return data_module.test_dataloader()
    else:
        assert False

def load_trained_supervised_model(
    model_dir: str, as_lightning_module: bool = False
):
    if model_dir.split('.')[-1] == 'h5': # this is the path of a pretrained Tensorflow model we have downloaded
        assert not as_lightning_module
        if 'cnn_best' in model_dir:
            from models.benadjila_models import CNNBest
            model = CNNBest((1, 700))
            model.load_pretrained_keras_params(model_dir)
        elif 'cnn2-ascad' in model_dir:
            from models.benadjila_models import CNNBest
            model = CNNBest((1, 1400))
            model.load_pretrained_keras_params(model_dir)
        elif 'mlp_best' in model_dir:
            from models.benadjila_models import MLPBest
            model = MLPBest((1, 700))
            model.load_pretrained_keras_params(model_dir)
        elif 'noConv1' in model_dir:
            seed = int(model_dir.split('_')[-1].split('.')[0])
            if 'aes_hd' in model_dir:
                from models.zaid_wouters_nets.pretrained_models import WoutersNet__AES_HD
                model = WoutersNet__AES_HD(pretrained_seed=seed)
            elif 'ascad' in model_dir:
                from models.zaid_wouters_nets.pretrained_models import WoutersNet__ASCADv1f
                model = WoutersNet__ASCADv1f(pretrained_seed=seed)
            elif 'dpav4' in model_dir:
                from models.zaid_wouters_nets.pretrained_models import WoutersNet__DPAv4
                model = WoutersNet__DPAv4(pretrained_seed=seed)
        elif 'zaid' in model_dir:
            seed = int(model_dir.split('_')[-1].split('.')[0])
            if 'aes_hd' in model_dir:
                from models.zaid_wouters_nets.pretrained_models import ZaidNet__AES_HD
                model = ZaidNet__AES_HD(pretrained_seed=seed)
            elif 'ascad' in model_dir:
                from models.zaid_wouters_nets.pretrained_models import ZaidNet__ASCADv1f
                model = ZaidNet__ASCADv1f(pretrained_seed=seed)
            elif 'dpav4' in model_dir:
                from models.zaid_wouters_nets.pretrained_models import ZaidNet__DPAv4
                model = ZaidNet__DPAv4(pretrained_seed=seed)
            else:
                assert False
        else:
            raise NotImplementedError
        return model
    else: # this is a model we have trained
        checkpoint_path = os.path.join(model_dir, 'early_stop_checkpoint.ckpt')
        assert os.path.exists(checkpoint_path)
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
    reference_leakage_assessment: Optional[np.ndarray] = None,
    dataset_name: Optional[str] = None
):
    if training_kwargs is None:
        training_kwargs = {}
    trainer = SupervisedTrainer(
        profiling_dataset, attack_dataset, default_training_module_kwargs=training_kwargs, reference_leakage_assessment=reference_leakage_assessment, dataset_name=dataset_name
    )
    set_seed(seed)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    trainer.run(output_dir, max_steps=max_steps, plot_metrics_over_time=reference_leakage_assessment is not None)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    if not os.path.exists(os.path.join(output_dir, 'training_time.npy')):
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
    if not os.path.exists(os.path.join(output_dir, 'results.pickle')):
        if training_kwargs is None:
            training_kwargs = {}
        trainer = SupervisedTrainer(profiling_dataset, attack_dataset, default_training_module_kwargs=training_kwargs)
        trainer.hparam_tune(output_dir, trial_count=trial_count, max_steps=max_steps, starting_seed=starting_seed)

def get_best_supervised_model_hparams(sweep_dir: str, profiling_dataset: Dataset, attack_dataset: Dataset, dataset_name: str, reference_leakage_assessment: np.ndarray):
    best_model_dir = None
    best_val_rank, best_val_loss = float('inf'), float('inf')
    trial_count = max(int(x.split('_')[-1]) for x in os.listdir(sweep_dir) if x.split('_')[0] == 'trial')
    dirpaths = [os.path.join(sweep_dir, f'trial_{x}') for x in range(trial_count)]
    print('Finding best supervised model...')
    results = defaultdict(list)
    for model_dir in tqdm(dirpaths):
        assert 'training_curves.pickle' in os.listdir(os.path.join(sweep_dir, model_dir))
        assert 'early_stop_checkpoint.ckpt' in os.listdir(os.path.join(sweep_dir, model_dir))
        with open(os.path.join(sweep_dir, model_dir, 'training_curves.pickle'), 'rb') as f:
            training_curves = pickle.load(f)
        val_rank_over_time = training_curves['val_rank'][-1]
        val_loss_over_time = training_curves['val_loss'][-1]
        early_stop_idx = np.argmin(val_rank_over_time)
        val_rank = val_rank_over_time[early_stop_idx]
        val_loss = val_loss_over_time[early_stop_idx]
        if (val_rank < best_val_rank) or (val_rank == best_val_rank and val_loss < best_val_loss):
            best_model_dir = model_dir
            best_val_rank = val_rank
            best_val_loss = val_loss
        attribute_neural_net(
            model_dir, profiling_dataset, attack_dataset, dataset_name,
            compute_gradvis=True, compute_saliency=True, compute_inputxgrad=True, compute_lrp=True, compute_occlusion=[1]
        )
        def record_result(key):
            assessment = np.load(os.path.join(model_dir, f'{key}.npz'))['attribution']
            if assessment.std() == 0:
                results[key].append(0.)
            else:
                results[key].append(spearmanr(assessment, reference_leakage_assessment).statistic)
        for key in ['gradvis', 'saliency', 'lrp', 'inputxgrad', '1-occlusion']:
            record_result(key)
    fig, axes = plt.subplots(1, 5, figsize=(5*PLOT_WIDTH, PLOT_WIDTH))
    for (key, val), ax in zip(results.items(), axes):
        ax.hist(val, color='blue')
        ax.set_label('Oracle agreement')
        ax.set_ylabel('Count')
        ax.set_title(key)
    fig.tight_layout()
    fig.savefig(os.path.join(sweep_dir, 'oracle_agreement_histograms.png'))
    plt.close(fig)

    best_indices = {
        'classification': int(best_model_dir.split('_')[-1]),
        **{
            f'oracle_{method}': np.argmax(results[method]) for method in ['gradvis', 'saliency', 'lrp', 'inputxgrad', '1-occlusion']
        }
    }
    best_hparams = {}
    for name, idx in best_indices.items():
        with open(os.path.join(sweep_dir, f'trial_{idx}', 'hparams.json'), 'r') as f:
            hparams = json.load(f)
        best_hparams[name] = hparams

    return best_hparams

# Create plots showing the performance of a trained supervised model
def eval_on_attack_dataset(model_dir: str, profile_dataset: Dataset, attack_dataset: Dataset, dataset_name: str, output_dir: Optional[str] = None):
    if output_dir is None:
        output_dir = model_dir
    os.makedirs(output_dir, exist_ok=True)
    if dataset_name in ['ascadv1-fixed', 'ascadv1-variable', 'aes-hd', 'dpav4']:
        if os.path.exists(os.path.join(output_dir, 'attack_performance.npy')):
            rv = np.load(os.path.join(output_dir, 'attack_performance.npy'))
        else:
            set_seed(0)
            dataloader = get_dataloader(
                profile_dataset, attack_dataset, split='attack',
                **({'data_mean': torch.tensor(0.0), 'data_var': torch.tensor(1.0)} if any(x in model_dir for x in ['cnn_best', 'mlp_best', 'cnn2-ascad']) else {})) # Benadjila models don't seem to have normalized data for training
            model = load_trained_supervised_model(model_dir, as_lightning_module=False)
            evaluator = AESMultiTraceEvaluator(dataloader, model, seed=0, dataset_name=dataset_name)
            rv = evaluator(get_rank_over_time=True)
            np.save(os.path.join(output_dir, 'attack_performance.npy'), rv)
        fig, ax = plt.subplots(1, 1, figsize=(PLOT_WIDTH, PLOT_WIDTH))
        ax.plot(np.arange(1, len(rv)+1), rv, color='blue')
        ax.set_xlabel('Traces seen')
        ax.set_ylabel('Rank of correct key')
        ax.set_xscale('log')
        ax.set_yscale('log')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'attack_performance.png'))
        plt.close(fig)
    if os.path.exists(os.path.join(output_dir, 'attack_rank_and_loss.npy')):
        loss, rank = np.load(os.path.join(output_dir, 'attack_rank_and_loss.npy'))
    else:
        dataloader = get_dataloader(profile_dataset, attack_dataset, split='attack')
        model = load_trained_supervised_model(model_dir, as_lightning_module=False)
        evaluator = AESMultiTraceEvaluator(dataloader, model, seed=0, dataset_name=dataset_name)
        loss, rank = evaluator(get_rank_over_time=False)
        np.save(os.path.join(output_dir, 'attack_rank_and_loss.npy'), np.array([loss, rank]))

# Compute various neural net attribution leakage assessments given a trained model directory
def attribute_neural_net(
    model_dir, profiling_dataset: Dataset, attack_dataset: Dataset, dataset_name: str,
    compute_gradvis: bool = False, compute_saliency: bool = False, compute_inputxgrad: bool = False,
    compute_lrp: bool = False, compute_occlusion: List[int] = [], compute_second_order_occlusion: List[int] = [],
    compute_occpoi: bool = False, compute_extended_occpoi: bool = False, output_dir: str = None
):
    if output_dir is None:
        output_dir = model_dir
    profiling_dataloader = None
    attack_dataloader = None
    model = None
    neural_net_attributor = None
    occpoi_computor = None
    def init(mode: Literal['occpoi', 'attr'] = 'attr'): # since these take time to init and we don't want to do it if not needed
        nonlocal profiling_dataloader, attack_dataloader, model, neural_net_attributor, occpoi_computor
        profiling_dataloader = profiling_dataloader or get_dataloader(profiling_dataset, attack_dataset, split='profile')
        attack_dataloader = attack_dataloader or get_dataloader(attack_dataset, attack_dataset, split='profile')
        model = model or load_trained_supervised_model(model_dir, as_lightning_module=False)
        if mode == 'attr':
            neural_net_attributor = neural_net_attributor or NeuralNetAttribution(profiling_dataloader, model, seed=0, device='cuda' if torch.cuda.is_available() else 'cpu')
        elif mode == 'occpoi':
            occpoi_computor = occpoi_computor or OccPOI(attack_dataloader, model, seed=0, device='cuda' if torch.cuda.is_available() else 'cpu', dataset_name=dataset_name)
        else:
            assert False
    def compute_attribution(attribution_fn: Callable, filename: str, mode: Literal['attr', 'occpoi'] = 'attr'):
        if os.path.exists(os.path.join(output_dir, filename)):
            return
        init(mode=mode)
        if not os.path.exists(os.path.join(output_dir, filename)):
            set_seed(0)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            attribution = attribution_fn()
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
            rv = {'attribution': attribution, 'elapsed_time': elapsed_time}
            np.savez(os.path.join(output_dir, filename), **rv)
    if compute_gradvis:
        compute_attribution(lambda: neural_net_attributor.compute_gradvis(), 'gradvis.npz')
    if compute_saliency:
        compute_attribution(lambda: neural_net_attributor.compute_saliency(), 'saliency.npz')
    if compute_inputxgrad:
        compute_attribution(lambda: neural_net_attributor.compute_inputxgrad(), 'inputxgrad.npz')
    if compute_lrp:
        compute_attribution(lambda: neural_net_attributor.compute_lrp(), 'lrp.npz')
    for window_size in compute_occlusion:
        compute_attribution(lambda: neural_net_attributor.compute_n_occlusion(window_size), f'{window_size}-occlusion.npz')
    for window_size in compute_second_order_occlusion:
        compute_attribution(lambda: neural_net_attributor.compute_second_order_occlusion(window_size=window_size), f'{window_size}-second-order-occlusion.npz')
    if compute_occpoi:
        compute_attribution(lambda: occpoi_computor(extended=False), 'occpoi.npz', mode='occpoi')
    if False: #compute_extended_occpoi:
        compute_attribution(lambda: occpoi_computor(extended=True), 'extended-occpoi.npz')

def evaluate_model_performance(
    base_dir: str
):
    print(f'Computing attack performance for models in {base_dir}')
    model_dirs = [os.path.join(base_dir, x) for x in os.listdir(base_dir) if x.split('=')[0] == 'seed']
    if all(os.path.exists(os.path.join(x, 'attack_performance.npy')) for x in model_dirs):
        rank_over_time_curves = np.stack([
            np.load(os.path.join(x, 'attack_performance.npy')) for x in model_dirs if os.path.exists(os.path.join(x, 'attack_performance.npy'))
        ])
        traces_to_disclosure = np.stack([
            np.nonzero(x-1)[0][-1] + 1 if len(np.nonzero(x-1)) > 0 else 1 for x in rank_over_time_curves
        ])
        print(f'\tTraces to AES key disclosure: {traces_to_disclosure.mean()} +/- {traces_to_disclosure.std()}')
    losses_and_ranks = np.stack([
        np.load(os.path.join(x, 'attack_rank_and_loss.npy')) for x in model_dirs
    ])
    losses = losses_and_ranks[:, 0]
    ranks = losses_and_ranks[:, 1]
    print(f'\tLoss: {losses.mean()} +/- {losses.std()}')
    print(f'\tRanks: {ranks.mean()} +/- {ranks.std()}')

def evaluate_leakage_assessments(
    base_dir: str, oracle_assessment: np.ndarray
):
    attr_filenames =  ['gradvis', 'saliency', 'inputxgrad', 'lrp', 'occpoi', 'extended-occpoi', '1-occlusion', '1-second-order-occlusion']
    model_dirs = [os.path.join(base_dir, x) for x in os.listdir(base_dir) if x.split('=')[0] == 'seed']
    assessments = defaultdict(list)
    elapsed_times = defaultdict(list)
    assessments_over_time = defaultdict(list)
    model_training_times = []
    for model_dir in model_dirs:
        model_training_times.append(np.load(os.path.join(model_dir, 'training_time.npy')))
        for attr_filename in attr_filenames:
            if os.path.exists(os.path.join(model_dir, f'{attr_filename}.npz')):
                res = np.load(os.path.join(model_dir, f'{attr_filename}.npz'), allow_pickle=True)
                assessment = res['attribution']
                elapsed_time = res['elapsed_time']
                assessments[attr_filename].append(assessment)
                elapsed_times[attr_filename].append(elapsed_time)
        #with open(os.path.join(model_dir, 'training_curves.pickle'), 'rb') as f:
        #    training_curves = pickle.load(f)
        #for method in ['gradvis', 'inputxgrad', 'lrp', 'saliency', '1-occlusion']:
        #    assessments_over_time[method].append(training_curves[f'{method}_oracle_agreement'][1])
    assessments = {key: np.stack(val) for key, val in assessments.items()}
    elapsed_times = {key: np.stack(val) for key, val in elapsed_times.items()}
    model_training_times = np.array(model_training_times)
    assessments_over_time = {key: np.stack(val) for key, val in assessments_over_time.items()}
    print(f'Base model training time: {model_training_times.mean()} +/- {model_training_times.std()}')
    for attr_filename in assessments:
        osnr_scores = [spearmanr(x, oracle_assessment).statistic for x in assessments[attr_filename]]
        osnr_scores = np.array([0. if np.isnan(x) else x for x in osnr_scores])
        times = 1e-3*elapsed_times[attr_filename]/60
        print(f'Method: {attr_filename}')
        print(f'\tOracle agreement: {osnr_scores.mean()} +/- {osnr_scores.std()}')
        print(f'\tElapsed time: {times.mean()} +/- {times.std()}')

    fig, axes = plt.subplots(2, len(assessments)//2, figsize=(len(assessments)*PLOT_WIDTH//2, 2*PLOT_WIDTH))
    for ax, (assessment_name, assessment) in zip(axes.flatten(), assessments.items()):
        for _assessment in assessment:
            ax.plot(oracle_assessment, _assessment, marker='.', linestyle='none', markersize=1)
        ax.set_xlabel('Oracle leakiness')
        ax.set_ylabel('Estimated leakiness')
        ax.set_title(assessment_name)
        ax.set_xscale('log')
        ax.set_yscale('log')
    fig.savefig(os.path.join(base_dir, 'assessments_vs_oracle.png'))
    plt.close(fig)

    fig, axes = plt.subplots(2, len(assessments)//2, figsize=(len(assessments)*PLOT_WIDTH//2, 2*PLOT_WIDTH))
    for ax, (assessment_name, assessment) in zip(axes.flatten(), assessments.items()):
        ax.fill_between(np.arange(assessment.shape[1]), assessment.mean(axis=0)-assessment.std(axis=0), assessment.mean(axis=0)+assessment.std(axis=0), color='blue', alpha=0.25)
        ax.plot(assessment.mean(axis=0), marker='.', markersize=1, linestyle='none', color='blue')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Estimated leakiness')
        ax.set_title(assessment_name)
    fig.tight_layout()
    fig.savefig(os.path.join(base_dir, 'assessments.png'))
    plt.close(fig)

    r"""fig, ax = plt.subplots(1, 1, figsize=(PLOT_WIDTH, PLOT_WIDTH))
    colors = ['red', 'green', 'blue', 'purple']
    for (assessment_name, assessment), color in zip(assessments_over_time.items(), colors):
        mean, std = assessment.mean(axis=0), assessment.std(axis=0)
        ax.fill_between(np.arange(len(mean)), mean-std, mean+std, alpha=0.25, color=color)
        ax.plot(mean, marker='.', linestyle='none', color=color, label=assessment_name)
    ax.set_xlabel('Training steps')
    ax.set_ylabel('Agreement with oracle')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(base_dir, 'assessments_over_time.png'))
    plt.close(fig)"""