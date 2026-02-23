import numpy as np
import torch
from torch import nn

from leakage_localization.utils import aes

class BitLogitsToByteLogits(nn.Module):
    bits: torch.Tensor

    def __init__(self):
        super().__init__()
        byte_values = torch.arange(256)
        bits = ((byte_values.unsqueeze(1) >> torch.arange(8)) & 1).to(torch.float32)
        self.register_buffer('bits', bits, persistent=False)
    
    def forward(self, bit_logits: torch.Tensor) -> torch.Tensor:
        *batch_dims, bit_count = bit_logits.shape
        assert bit_count == 8
        bit_logits = bit_logits.reshape(-1, bit_count)
        logp1 = nn.functional.logsigmoid(bit_logits)
        logp0 = nn.functional.logsigmoid(-bit_logits)
        logp_byte = (self.bits*logp1.unsqueeze(1) + (1-self.bits)*logp0.unsqueeze(1)).sum(dim=-1)
        logp_byte = logp_byte - torch.logsumexp(logp_byte, dim=-1, keepdim=True)
        logp_byte = logp_byte.reshape(*batch_dims, 256)
        return logp_byte

class ByteLogitsToBitLogits(nn.Module):
    bits: torch.Tensor

    def __init__(self):
        super().__init__()
        byte_values = torch.arange(256)
        bits = ((byte_values.unsqueeze(1) >> torch.arange(8)) & 1).to(torch.bool)
        self.register_buffer('bits', bits, persistent=False)
    
    def forward(self, byte_logits: torch.Tensor) -> torch.Tensor:
        *batch_dims, id_count = byte_logits.shape
        assert id_count == 256
        byte_logits = byte_logits.reshape(-1, id_count)
        logp = torch.log_softmax(byte_logits, dim=-1)
        logp_masked1 = torch.where(self.bits.unsqueeze(0), logp.unsqueeze(2), float('-inf'))
        logp_masked0 = torch.where(~self.bits.unsqueeze(0), logp.unsqueeze(2), float('-inf'))
        logp1 = torch.logsumexp(logp_masked1, dim=-2)
        logp0 = torch.logsumexp(logp_masked0, dim=-2)
        bit_logits = logp1 - logp0
        bit_logits = bit_logits.reshape(*batch_dims, 8)
        return bit_logits

class ByteLogitsToHwLogits(nn.Module):
    hw_mask: torch.Tensor

    def __init__(self):
        super().__init__()
        byte_values = torch.arange(256)
        bits = ((byte_values.unsqueeze(1) >> torch.arange(8)) & 1)
        hw = bits.sum(dim=1)
        hw_mask = torch.stack([(hw == i) for i in range(9)], dim=0)
        self.register_buffer('hw_mask', hw_mask, persistent=False)
    
    def forward(self, byte_logits: torch.Tensor) -> torch.Tensor:
        *batch_dims, id_count = byte_logits.shape
        assert id_count == 256
        byte_logits = byte_logits.reshape(-1, id_count)
        log_byte_probs = torch.log_softmax(byte_logits, dim=-1)
        log_hw_probs = torch.where(self.hw_mask.unsqueeze(0), log_byte_probs.unsqueeze(1), float('-inf'))
        log_hw_probs = torch.logsumexp(log_hw_probs, dim=-1)
        log_hw_probs = log_hw_probs.reshape(*batch_dims, 9)
        return log_hw_probs

class SoftXOR(nn.Module):
    xor_lut: torch.Tensor

    def __init__(self):
        super().__init__()
        byte_values = torch.arange(256)
        xor_lut = byte_values.unsqueeze(0) ^ byte_values.unsqueeze(1)
        self.register_buffer('xor_lut', xor_lut, persistent=False)
    
    def forward(self, x_logits: torch.Tensor, y_logits: torch.Tensor) -> torch.Tensor:
        logpx = torch.log_softmax(x_logits, dim=-1)
        logpy = torch.log_softmax(y_logits, dim=-1)
        gathered_logpy = logpy[..., self.xor_lut]
        log_joint = logpx.unsqueeze(-1) + gathered_logpy
        logpz = torch.logsumexp(log_joint, dim=-2)
        return logpz

class SoftGF256Mult(nn.Module):
    invgf256mult_lut: torch.Tensor

    def __init__(self):
        super().__init__()
        byte_values = np.arange(256, dtype=np.uint8)
        gf256mult_lut = aes.mult_gf256(byte_values[np.newaxis, :], byte_values[:, np.newaxis])
        invgf256mult_lut = np.zeros((255, 256), dtype=np.int64)
        for a in range(1, 256):
            invgf256mult_lut[a-1, gf256mult_lut[a, :]] = byte_values
        self.register_buffer('invgf256mult_lut', torch.from_numpy(invgf256mult_lut), persistent=False)
    
    def forward(self, x_logits: torch.Tensor, y_logits: torch.Tensor) -> torch.Tensor:
        logpx = torch.log_softmax(x_logits, dim=-1)
        logpx_nonzero = logpx[..., 1:]
        logpy = torch.log_softmax(y_logits, dim=-1)
        gathered_logpy = logpy[..., self.invgf256mult_lut]
        log_joint = logpx_nonzero.unsqueeze(-1) + gathered_logpy
        logpz = torch.logsumexp(log_joint, dim=-2)
        logpz_zero = torch.logsumexp(torch.stack([logpz[..., 0:1], logpx[..., 0:1]]), dim=0)
        logpz = torch.cat([logpz_zero, logpz[..., 1:]], dim=-1)
        return logpz