from typing import Dict, Any, Literal
from math import log, log1p

import torch
from torch import nn
from reinmax import reinmax

import models

class CondMutInfEstimator(nn.Module):
    def __init__(self,
        timesteps_per_trace: int,
        output_classes: int,
        classifiers_name: str,
        classifiers_kwargs: Dict[str, Any] = {},
        omit_classifier_conditioning: bool = False
    ):
        super().__init__()
        self.timesteps_per_trace = timesteps_per_trace
        self.output_classes = output_classes
        self.classifiers_name = classifiers_name
        self.classifiers_kwargs = classifiers_kwargs
        self.omit_classifier_conditioning = omit_classifier_conditioning

        self.classifiers = models.load(
            self.classifiers_name,
            input_shape=(1, self.timesteps_per_trace),
            output_classes=self.output_classes,
            noise_conditional=not self.omit_classifier_conditioning,
            **self.classifiers_kwargs
        )
    
    def get_logits(self, input: torch.Tensor, condition_mask: torch.Tensor) -> torch.Tensor:
        assert input.shape == condition_mask.shape
        batch_size, *input_shape = input.shape
        assert tuple(input_shape) == (1, self.timesteps_per_trace)
        masked_input = condition_mask*input + (1-condition_mask)*torch.randn_like(input)
        if self.omit_classifier_conditioning:
            logits = self.classifiers(masked_input)
        else:
            logits = self.classifiers(masked_input, condition_mask)
        logits = logits.reshape(batch_size, self.output_classes)
        return logits
    
    def get_mutinf_estimate_from_logits(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size, class_count = logits.shape
        assert class_count == self.output_classes
        ent_y = torch.full((batch_size,), log(self.output_classes), dtype=logits.dtype, device=logits.device)
        ent_y_mid_x = nn.functional.cross_entropy(logits, labels)
        mutinf = ent_y - ent_y_mid_x
        return mutinf
    
    def get_mutinf_estimate(self, input: torch.Tensor, condition_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = self.get_logits(input, condition_mask)
        mutinf = self.get_mutinf_estimate_from_logits(logits, labels)
        return mutinf
    
    def forward(self, *args, **kwargs):
        assert False

class SelectionMechanism(nn.Module):
    def __init__(self, timesteps_per_trace: int, gamma_bar: float = 0.5, adversarial_mode: bool = True, relaxation_temp: float = 1.0, use_budget: bool = True):
        super().__init__()
        self.timesteps_per_trace = timesteps_per_trace
        self.gamma_bar = gamma_bar
        self.adversarial_mode = adversarial_mode
        self.relaxation_temp = relaxation_temp
        self.use_budget = use_budget
        if self.use_budget:
            log_C = log(self.timesteps_per_trace) + log(gamma_bar) - log1p(-gamma_bar)
            self.register_buffer('log_C', torch.tensor(log_C))
        self.etat = nn.Parameter(0.01*torch.randn((1, self.timesteps_per_trace)))
    
    def get_etat(self) -> torch.Tensor:
        return self.etat
    
    def get_eta(self) -> torch.Tensor:
        etat = self.get_etat()
        eta = nn.functional.softmax(etat, dim=-1)
        return eta
    
    def get_log_eta(self) -> torch.Tensor:
        etat = self.get_etat()
        log_eta = nn.functional.log_softmax(etat, dim=-1)
        return log_eta
    
    def get_gammat(self) -> torch.Tensor:
        etat = self.get_etat()
        if self.use_budget:
            gammat = etat + self.log_C - torch.logsumexp(etat.squeeze(0), dim=0)
        else:
            gammat = etat
        return gammat
    
    def get_gamma(self) -> torch.Tensor:
        gammat = self.get_gammat()
        gamma = nn.functional.sigmoid(gammat)
        return gamma
    
    def get_log_gamma(self) -> torch.Tensor:
        gammat = self.get_gammat()
        log_gamma = nn.functional.logsigmoid(gammat)
        return log_gamma
    
    def get_log_1mgamma(self) -> torch.Tensor:
        gammat = self.get_gammat()
        log_1mgamma = nn.functional.logsigmoid(-gammat)
        return log_1mgamma
    
    @torch.no_grad()
    def hard_sample(self, batch_size: int) -> torch.Tensor:
        gamma = self.get_gamma()
        probs = gamma.unsqueeze(0).repeat(batch_size, 1, 1)
        alpha = probs.bernoulli_()
        if self.adversarial_mode:
            alpha = 1 - alpha
        return alpha
    
    def log_pmf(self, alpha: torch.Tensor) -> torch.Tensor:
        log_gamma = self.get_log_gamma()
        log_1mgamma = self.get_log_1mgamma()
        log_pdf = (alpha*log_gamma + (1-alpha)*log_1mgamma).sum(-1).squeeze(-1)
        return log_pdf
    
    def reinmax_sample(self, batch_size: int) -> torch.Tensor:
        log_gamma = self.get_log_gamma().unsqueeze(0).repeat(batch_size, 1, 1).reshape(batch_size*self.timesteps_per_trace, 1)
        log_1mgamma = self.get_log_1mgamma().unsqueeze(0).repeat(batch_size, 1, 1).reshape(batch_size*self.timesteps_per_trace, 1)
        alpha, _ = reinmax(torch.stack([log_1mgamma, log_gamma], dim=-1), self.relaxation_temp)
        alpha = alpha[..., 1]
        alpha = alpha.reshape(batch_size, 1, self.timesteps_per_trace)
        if self.adversarial_mode:
            alpha = 1 - alpha
        return alpha
    
    def concrete_sample(self, batch_size: int) -> torch.Tensor:
        log_gamma = self.get_log_gamma().unsqueeze(0).repeat(batch_size, 1, 1)
        log_1mgamma = self.get_log_1mgamma().unsqueeze(0).repeat(batch_size, 1, 1)
        u = torch.rand_like(log_gamma).clamp_(min=1e-6, max=1-1e-6)
        z = log_gamma - log_1mgamma + u.log() - (1-u).log()
        alpha = nn.functional.sigmoid(z/self.relaxation_temp)
        if self.adversarial_mode:
            alpha = 1-alpha
        return alpha
    
    def forward(self, *args, **kwargs):
        assert False