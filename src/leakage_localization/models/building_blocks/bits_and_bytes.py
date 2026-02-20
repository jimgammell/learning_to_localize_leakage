import torch
from torch import nn

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
        logp = nn.functional.log_softmax(byte_logits, dim=-1)
        logp_masked1 = torch.where(self.bits.unsqueeze(0), logp.unsqueeze(2), float('-inf'))
        logp_masked0 = torch.where(~self.bits.unsqueeze(0), logp.unsqueeze(2), float('-inf'))
        logp1 = torch.logsumexp(logp_masked1, dim=-2)
        logp0 = torch.logsumexp(logp_masked0, dim=-2)
        bit_logits = logp1 - logp0
        bit_logits = bit_logits.reshape(*batch_dims, 8)
        return bit_logits