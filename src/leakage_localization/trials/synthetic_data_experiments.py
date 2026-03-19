from typing import *
import os
from numbers import Number
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrow

from common import *
from trials.utils import *
from utils.baseline_assessments import FirstOrderStatistics
from datasets.synthetic_aes import SyntheticAES, SyntheticAESLike
from training_modules.adversarial_leakage_localization import ALLTrainer

GAMMA_BAR_VALS = np.array([0.5])

class Trial:
    def __init__(self,
        logging_dir: Union[str, os.PathLike] = None,
        override_run_kwargs: dict = {},
        override_leakage_localization_kwargs: dict = {},
        batch_size: int = 1000,
        timestep_count: int = 101,
        trial_count: int = 8,
        seed_count: int = 1,
        pretrain_classifiers_only: bool = False
    ):
        self.logging_dir = logging_dir
        self.run_kwargs = {'max_steps': 10000, 'anim_gammas': False}
        self.leakage_localization_kwargs = {'classifiers_name': 'mlp-1d', 'theta_lr': 1e-3, 'theta_weight_decay': 1e-2, 'etat_lr': 1e-3, 'gamma_bar': 0.5}
        self.run_kwargs.update(override_run_kwargs)
        self.leakage_localization_kwargs.update(override_leakage_localization_kwargs)
        self.batch_size = batch_size
        self.timestep_count = timestep_count
        self.trial_count = trial_count
        self.seed_count = seed_count
        self.pretrain_classifiers_only = pretrain_classifiers_only
        self.betas = np.array([1 - 0.5**n for n in range(self.trial_count)][::-1])
        self.leaky_pt_counts = np.array([0] + [1 + 2*x for x in range(self.trial_count-1)])
        self.no_op_counts = np.array([0] + [1 + 4*x for x in range(self.trial_count-1)])
        self.shuffle_loc_counts = np.array([1 + 2*x for x in range(self.trial_count)])
    
    def construct_datasets(self,
        leaky_1o_count: int = 1,
        leaky_2o_count: int = 0,
        data_var: float = 1.0,
        shuffle_locs: int = 1,
        max_no_ops: int = 0,
        lpf_beta: float = 0.5   
    ):
        leaky_count = shuffle_locs*(leaky_1o_count + 2*leaky_2o_count)
        if leaky_count > 0:
            leaky_pts = np.linspace(0, self.timestep_count-1, leaky_count+2)[1:-1].astype(int)
            leaky_1o_pts = leaky_pts[:shuffle_locs*leaky_1o_count] if leaky_1o_count > 0 else None
            leaky_2o_pts = leaky_pts[shuffle_locs*leaky_1o_count:].reshape(2, -1) if leaky_2o_count > 0 else None
        else:
            leaky_pts = leaky_1o_pts = leaky_2o_pts = None
        profiling_dataset = SyntheticAES(
            infinite_dataset=True,
            timesteps_per_trace=self.timestep_count,
            leaking_timestep_count_1o=0,
            leaking_timestep_count_2o=0,
            leaky_1o_pts=leaky_1o_pts,
            leaky_2o_pts=leaky_2o_pts,
            data_var=data_var,
            shuffle_locs=shuffle_locs,
            max_no_ops=max_no_ops,
            lpf_beta=lpf_beta
        )
        attack_dataset = SyntheticAESLike(profiling_dataset, fixed_key=0)
        return profiling_dataset, attack_dataset, leaky_1o_pts, leaky_2o_pts
    
    def construct_trainer(self, profiling_dataset, attack_dataset):
        trainer = ALLTrainer(
            profiling_dataset, attack_dataset,
            default_data_module_kwargs={'train_batch_size': self.batch_size},
            default_training_module_kwargs={**self.leakage_localization_kwargs}
        )
        return trainer
    
    def run_experiment(self, logging_dir, kwargs):
        leakage_assessments = {}
        os.makedirs(logging_dir, exist_ok=True)
        profiling_dataset, attack_dataset, locs_1o, locs_2o = self.construct_datasets(**kwargs)
        #if not os.path.exists(os.path.join(logging_dir, 'classifiers_pretrain', 'best_checkpoint.ckpt')):
        #    trainer = self.construct_trainer(profiling_dataset, attack_dataset) # classifier pretraining is independent of budget
        #    trainer.pretrain_classifiers(os.path.join(logging_dir, 'classifiers_pretrain'), max_steps=self.run_kwargs['max_steps'])
        for gamma_bar in GAMMA_BAR_VALS:
            if not os.path.exists(os.path.join(logging_dir, f'gamma_bar={gamma_bar}', 'leakage_assessments.npz')):
                self.leakage_localization_kwargs['gamma_bar'] = gamma_bar
                trainer = self.construct_trainer(profiling_dataset, attack_dataset)
                leakage_assessment = trainer.run(
                    os.path.join(logging_dir, f'gamma_bar={gamma_bar}'),
                    pretrained_classifiers_logging_dir=None, #os.path.join(logging_dir, 'classifiers_pretrain'),
                    **self.run_kwargs
                )
                np.savez(os.path.join(logging_dir, f'gamma_bar={gamma_bar}', 'leakage_assessments.npz'), leakage_assessment=leakage_assessment, locs_1o=locs_1o, locs_2o=locs_2o)
            else:
                data = np.load(os.path.join(logging_dir, f'gamma_bar={gamma_bar}', 'leakage_assessments.npz'), allow_pickle=True)
                leakage_assessment = data['leakage_assessment']
                locs_1o = data['locs_1o']
                locs_2o = data['locs_2o']
        return leakage_assessment, locs_1o, locs_2o
    
    def run_1o_beta_sweep(self):
        exp_dir = os.path.join(self.logging_dir, '1o_beta_sweep')
        leakage_assessments = {}
        for seed in range(self.seed_count):
            for beta in self.betas:
                subdir = os.path.join(exp_dir, f'seed={seed}', f'beta={beta}')
                leakage_assessments[1-beta], *_ = self.run_experiment(subdir, {'lpf_beta': beta})
    
    def plot_1o_beta_sweep(self, axes=None, subsample=None):
        full_plot = axes is None
        betas = self.betas[::-1] if subsample is None else self.betas[subsample][::-1]
        if full_plot:
            fig, axes = plt.subplots(5, self.trial_count, figsize=(0.75*self.trial_count*PLOT_WIDTH, 0.75*5*PLOT_WIDTH))
        else:
            axes = axes.reshape(1, len(betas))
        exp_dir = os.path.join(self.logging_dir, '1o_beta_sweep')
        for beta_idx, beta in enumerate(betas):
            for gamma_bar_idx, gamma_bar in enumerate(GAMMA_BAR_VALS):
                leakage_assessments = []
                ax = axes[gamma_bar_idx, beta_idx]
                for seed in range(self.seed_count):
                    subdir = os.path.join(exp_dir, f'seed={seed}', f'beta={beta}', f'gamma_bar={gamma_bar}')
                    assert os.path.exists(os.path.join(subdir, 'leakage_assessments.npz'))
                    data = np.load(os.path.join(subdir, 'leakage_assessments.npz'), allow_pickle=True)
                    leakage_assessment = data['leakage_assessment'].reshape(-1)
                    loc_1o = data['locs_1o'][0]
                    leakage_assessments.append(leakage_assessment)
                leakage_assessments = np.stack(leakage_assessments)
                ax.axvline(loc_1o, linestyle='--', color='black')
                ax.fill_between(np.arange(self.timestep_count), np.min(leakage_assessments, axis=0), np.max(leakage_assessments, axis=0), color='blue', alpha=0.25, **PLOT_KWARGS)
                ax.plot(np.arange(self.timestep_count), np.median(leakage_assessments, axis=0), color='blue', marker='.', markersize=5, linestyle='-', linewidth=0.1, **PLOT_KWARGS)
                if full_plot:
                    ax.set_xlabel(r'Timestep $t$')
                    ax.set_ylabel(r'Estimated leakage of $X_t$')
                ax.set_title(r'LPF $\beta$: $'+f'{beta}'+r'$', fontsize=18)
                ax.set_xlim(0, self.timestep_count-1)
                ax.set_ylim(0.0, 1.0)
        if full_plot:
            fig.tight_layout()
            fig.savefig(os.path.join(exp_dir, 'beta_sweep.png'))
    
    def run_1o_data_var_sweep(self):
        exp_dir = os.path.join(self.logging_dir, '1o_data_var_sweep')
        leakage_assessments = {}
        for seed in range(self.seed_count):
            for var in [1.0] + [0.5**(-2*n) for n in range(1, self.trial_count//2)] + [0.5**(2*n) for n in range(1, self.trial_count//2)] + [0.0]:
                subdir = os.path.join(exp_dir, f'seed={seed}', f'var={var}')
                leakage_assessments[var], *_ = self.run_experiment(subdir, {'data_var': var})
    
    def run_1o_leaky_pt_count_sweep(self):
        exp_dir = os.path.join(self.logging_dir, '1o_leaky_pt_sweep')
        leakage_assessments = {}
        locss = []
        for seed in range(self.seed_count):
            for count in [0] + [1 + 2*x for x in range(self.trial_count-1)]:
                subdir = os.path.join(exp_dir, f'seed={seed}', f'count={count}')
                leakage_assessments[count], locs, _ = self.run_experiment(subdir, {'leaky_1o_count': count})
                locss.append(locs)
                
    def plot_1o_leaky_pt_count_sweep(self, axes=None, subsample=None):
        full_plot = axes is None
        leaky_pt_counts = self.leaky_pt_counts if subsample is None else self.leaky_pt_counts[subsample]
        if full_plot:
            fig, axes = plt.subplots(5, self.trial_count, figsize=(0.75*self.trial_count*PLOT_WIDTH, 0.75*5*PLOT_WIDTH))
        else:
            axes = axes.reshape(1, len(leaky_pt_counts))
        exp_dir = os.path.join(self.logging_dir, '1o_leaky_pt_sweep')
        for leaky_pt_count_idx, leaky_pt_count in enumerate(leaky_pt_counts):
            for gamma_bar_idx, gamma_bar in enumerate(GAMMA_BAR_VALS):
                leakage_assessments = []
                ax = axes[gamma_bar_idx, leaky_pt_count_idx]
                for seed in range(self.seed_count):
                    subdir = os.path.join(exp_dir, f'seed={seed}', f'count={leaky_pt_count}', f'gamma_bar={gamma_bar}')
                    assert os.path.exists(os.path.join(subdir, 'leakage_assessments.npz'))
                    data = np.load(os.path.join(subdir, 'leakage_assessments.npz'), allow_pickle=True)
                    leakage_assessment = data['leakage_assessment'].reshape(-1)
                    locs_1o = data['locs_1o']
                    leakage_assessments.append(leakage_assessment)
                leakage_assessments = np.stack(leakage_assessments)
                try:
                    for loc_1o in locs_1o:
                        ax.axvline(loc_1o, linestyle='--', color='black')
                except TypeError:
                    pass
                ax.fill_between(np.arange(self.timestep_count), np.min(leakage_assessments, axis=0), np.max(leakage_assessments, axis=0), color='blue', alpha=0.25, **PLOT_KWARGS)
                ax.plot(np.arange(self.timestep_count), np.median(leakage_assessments, axis=0), color='blue', marker='.', markersize=5, linestyle='-', linewidth=0.1, **PLOT_KWARGS)
                if full_plot:
                    ax.set_xlabel(r'Timestep $t$')
                    ax.set_ylabel(r'Estimated leakage of $X_t$')
                ax.set_title(f'Leaky pt. cnt.: {leaky_pt_count}', fontsize=18)
                ax.set_xlim(0, self.timestep_count-1)
                ax.set_ylim(0.0, 1.0)
        if full_plot:
            fig.tight_layout()
            fig.savefig(os.path.join(exp_dir, 'leaky_pt_count_sweep.png'), **SAVEFIG_KWARGS)
    
    def run_1o_no_op_count_sweep(self):
        exp_dir = os.path.join(self.logging_dir, '1o_no_op_sweep')
        leakage_assessments = {}
        locss = []
        for seed in range(self.seed_count):
            for count in self.no_op_counts:
                subdir = os.path.join(exp_dir, f'seed={seed}', f'count={count}')
                leakage_assessments[count], locs, _ = self.run_experiment(subdir, {'max_no_ops': count})
                locss.append(locs)

    def plot_1o_no_op_count_sweep(self, axes=None, subsample=None):
        full_plot = axes is None
        no_op_counts = self.no_op_counts if subsample is None else self.no_op_counts[subsample]
        if full_plot:
            fig, axes = plt.subplots(5, self.trial_count, figsize=(0.75*self.trial_count*PLOT_WIDTH, 0.75*5*PLOT_WIDTH))
        else:
            axes = axes.reshape(1, len(no_op_counts))
        exp_dir = os.path.join(self.logging_dir, '1o_no_op_sweep')
        for no_op_count_idx, no_op_count in enumerate(no_op_counts):
            for gamma_bar_idx, gamma_bar in enumerate(GAMMA_BAR_VALS):
                leakage_assessments = []
                ax = axes[gamma_bar_idx, no_op_count_idx]
                for seed in range(self.seed_count):
                    subdir = os.path.join(exp_dir, f'seed={seed}', f'count={no_op_count}', f'gamma_bar={gamma_bar}')
                    assert os.path.exists(os.path.join(subdir, 'leakage_assessments.npz'))
                    data = np.load(os.path.join(subdir, 'leakage_assessments.npz'), allow_pickle=True)
                    leakage_assessment = data['leakage_assessment'].reshape(-1)
                    locs_1o = data['locs_1o']
                    leakage_assessments.append(leakage_assessment)
                leakage_assessments = np.stack(leakage_assessments)
                try:
                    for loc_1o in locs_1o:
                        ax.axvline(loc_1o, linestyle='--', color='black')
                except TypeError:
                    pass
                ax.fill_between(np.arange(self.timestep_count), np.min(leakage_assessments, axis=0), np.max(leakage_assessments, axis=0), color='blue', alpha=0.25, **PLOT_KWARGS)
                ax.plot(np.arange(self.timestep_count), np.median(leakage_assessments, axis=0), color='blue', marker='.', markersize=5, linestyle='-', linewidth=0.1, **PLOT_KWARGS)
                if full_plot:
                    ax.set_xlabel(r'Timestep $t$')
                    ax.set_ylabel(r'Estimated leakage of $X_t$')
                ax.set_title(f'Max no-op cnt.: {no_op_count}', fontsize=18)
                ax.set_xlim(0, self.timestep_count-1)
                ax.set_ylim(0.0, 1.0)
        if full_plot:
            fig.tight_layout()
            fig.savefig(os.path.join(exp_dir, 'no_op_count_sweep.png'), **SAVEFIG_KWARGS)
    
    def run_1o_shuffle_loc_sweep(self):
        exp_dir = os.path.join(self.logging_dir, '1o_shuffle_sweep')
        leakage_assessments = {}
        locss = []
        for seed in range(self.seed_count):
            for count in self.shuffle_loc_counts:
                subdir = os.path.join(exp_dir, f'seed={seed}', f'count={count}')
                leakage_assessments[count], locs, _ = self.run_experiment(subdir, {'shuffle_locs': count})
                locss.append(locs)

    def plot_1o_shuffle_loc_sweep(self, axes=None, subsample=None):
        full_plot = axes is None
        shuffle_loc_counts = self.shuffle_loc_counts if subsample is None else self.shuffle_loc_counts[subsample]
        if full_plot:
            fig, axes = plt.subplots(5, self.trial_count, figsize=(0.75*self.trial_count*PLOT_WIDTH, 0.75*5*PLOT_WIDTH))
        else:
            axes = axes.reshape(1, len(shuffle_loc_counts))
        exp_dir = os.path.join(self.logging_dir, '1o_shuffle_sweep')
        for shuffle_loc_count_idx, shuffle_loc_count in enumerate(shuffle_loc_counts):
            for gamma_bar_idx, gamma_bar in enumerate(GAMMA_BAR_VALS):
                leakage_assessments = []
                ax = axes[gamma_bar_idx, shuffle_loc_count_idx]
                for seed in range(self.seed_count):
                    subdir = os.path.join(exp_dir, f'seed={seed}', f'count={shuffle_loc_count}', f'gamma_bar={gamma_bar}')
                    assert os.path.exists(os.path.join(subdir, 'leakage_assessments.npz'))
                    data = np.load(os.path.join(subdir, 'leakage_assessments.npz'), allow_pickle=True)
                    leakage_assessment = data['leakage_assessment'].reshape(-1)
                    locs_1o = data['locs_1o']
                    leakage_assessments.append(leakage_assessment)
                leakage_assessments = np.stack(leakage_assessments)
                try:
                    for loc_1o in locs_1o:
                        ax.axvline(loc_1o, linestyle='--', color='black')
                except TypeError:
                    pass
                ax.fill_between(np.arange(self.timestep_count), np.min(leakage_assessments, axis=0), np.max(leakage_assessments, axis=0), color='blue', alpha=0.25, **PLOT_KWARGS)
                ax.plot(np.arange(self.timestep_count), np.median(leakage_assessments, axis=0), color='blue', marker='.', markersize=5, linestyle='-', linewidth=0.1, **PLOT_KWARGS)
                if full_plot:
                    ax.set_xlabel(r'Timestep $t$')
                    ax.set_ylabel(r'Estimated leakage of $X_t$')
                ax.set_title(f'Shuffle loc. cnt.: {shuffle_loc_count}', fontsize=18)
                ax.set_xlim(0, self.timestep_count-1)
                ax.set_ylim(0.0, 1.0)
        if full_plot:
            fig.tight_layout()
            fig.savefig(os.path.join(exp_dir, 'shuffle_loc_count_sweep.png'), **SAVEFIG_KWARGS)
    
    def plot_main_paper_sweeps(self, dest: str, dont_subsample=False):
        subsample = np.linspace(0, self.trial_count-1, 4).astype(int) if not dont_subsample else None
        trial_count = len(subsample) if subsample is not None else self.trial_count
        #subsample = np.arange(self.trial_count)
        fig, axes = plt.subplots(4, trial_count, figsize=(0.75*trial_count*PLOT_WIDTH, 0.75*4*PLOT_WIDTH))
        self.plot_1o_beta_sweep(axes[0, :], subsample)
        self.plot_1o_leaky_pt_count_sweep(axes[1, :], subsample)
        self.plot_1o_no_op_count_sweep(axes[2, :], subsample)
        self.plot_1o_shuffle_loc_sweep(axes[3, :], subsample)
        for ax in axes[:, 0]:
            ax.set_ylabel(r'Estimated leakage of $X_t$', fontsize=14)
        for ax in axes[-1, :]:
            ax.set_xlabel(r'Timestep $t$', fontsize=14)
        for ax in axes.flatten():
            ax.set_xlim(0, self.timestep_count-1)
            ax.set_ylim(-0.05, 1.05)
            ax.set_yticks([0.0, 1.0])
            ax.set_yticklabels(['0.', '1.'])
        fig.tight_layout()
        fig.canvas.draw_idle()
        fig.savefig(dest, **SAVEFIG_KWARGS)
    
    def run_2o_trial(self):
        exp_dir = os.path.join(self.logging_dir, '2o_trial')
        leakage_assessment, _, locs = self.run_experiment(exp_dir, {'leaky_1o_count': 0, 'leaky_2o_count': 1})
        self.plot_leakage_assessments(
            os.path.join(exp_dir, 'sweep.pdf'),
            {None: leakage_assessment},
            locs,
            title=r'Second-order leakage'
        )
    
    def __call__(self):
        self.run_1o_beta_sweep()
        self.plot_1o_beta_sweep()
        self.run_1o_leaky_pt_count_sweep()
        self.plot_1o_leaky_pt_count_sweep()
        self.run_1o_no_op_count_sweep()
        self.plot_1o_no_op_count_sweep()
        self.run_1o_shuffle_loc_sweep()
        self.plot_1o_shuffle_loc_sweep()
        self.plot_main_paper_sweeps(os.path.join(self.logging_dir, 'main_paper_sweep.pdf'))
        self.plot_main_paper_sweeps(os.path.join(self.logging_dir, 'full_synthetic_sweeps_for_appendix.pdf'), dont_subsample=True)