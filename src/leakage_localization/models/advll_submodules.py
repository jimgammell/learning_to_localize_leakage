from math import log, log1p

import torch
from torch import nn

class SelectionMechanism(nn.Module):
    log_C: torch.Tensor

    def __init__(
            self,
            in_features: int,
            gamma_bar: float,
            relaxation_temp: float = 1.
    ):
        super().__init__()
        
        self.in_features = in_features
        self.gamma_bar = gamma_bar
        self.relaxation_temp = relaxation_temp

        log_C = log(self.in_features) + log(self.gamma_bar) - log1p(-self.gamma_bar)
        self.register_buffer('log_C', torch.tensor(log_C))
        self.etat = nn.Parameter(0.01*torch.randn((1, self.in_features)))
    
    def get_etat(self) -> torch.Tensor:
        return self.etat
    
    def get_eta(self) -> torch.Tensor:
        etat = self.get_etat()
        eta = torch.softmax(etat, dim=-1)
        return eta
    
    def get_log_eta(self) -> torch.Tensor: # using log_softmax is more numerically-stable than log(softmax)
        etat = self.get_etat()
        log_eta = torch.log_softmax(etat, dim=-1)
        return log_eta
    
    def get_gammat(self) -> torch.Tensor:
        etat = self.get_etat()
        gammat = etat + self.log_C - torch.logsumexp(etat.squeeze(0), dim=0)
        return gammat
    
    def get_gamma(self) -> torch.Tensor:
        gammat = self.get_gammat()
        gamma = torch.sigmoid(gammat)
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
        alpha = 1 - probs.bernoulli_()
        return alpha
    
    def log_pmf(self, alpha: torch.Tensor) -> torch.Tensor:
        log_gamma = self.get_log_gamma()
        log_1mgamma = self.get_log_1mgamma()
        log_pdf = (alpha*log_gamma + (1-alpha)*log_1mgamma).sum(dim=-1).squeeze(-1)
        return log_pdf
    
    def concrete_sample(self, batch_size: int) -> torch.Tensor:
        log_gamma = self.get_log_gamma().unsqueeze(0).repeat(batch_size, 1, 1)
        log_1mgamma = self.get_log_1mgamma().unsqueeze(0).repeat(batch_size, 1, 1)
        u = torch.rand_like(log_gamma).clamp_(min=1.e-6, max=1-1.e-6)
        z = log_gamma - log_1mgamma + u.log() - (1-u).log()
        alpha = 1 - torch.sigmoid(z/self.relaxation_temp)
        return alpha
    
    def forward(self, *args, **kwargs):
        assert False