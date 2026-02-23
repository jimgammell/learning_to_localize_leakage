import torch
from torch import nn

from .bits_and_bytes import SoftXOR, SoftSbox, SoftGF256Mult, SoftGF256Inv

class ASCADv1Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.soft_xor = SoftXOR()
        self.soft_sbox = SoftSbox()
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        batch_size, label_count, class_count = logits.shape
        assert label_count == 66
        assert class_count == 256
        r_logits = logits[:, 0:16, :]
        rin_logits = logits[:, 16:17, :]
        rout_logits = logits[:, 17:18, :]
        kwrin_logits = logits[:, 18:34, :]
        rsbox_logits = logits[:, 34:50, :]
        routsbox_logits = logits[:, 50:66, :]
        sbox_logits_0 = self.soft_xor(r_logits, rsbox_logits)
        sbox_logits_1 = self.soft_sbox(self.soft_xor(rin_logits, kwrin_logits))
        sbox_logits_2 = self.soft_xor(rout_logits, routsbox_logits)
        sbox_logits = torch.log_softmax(torch.stack([sbox_logits_0, sbox_logits_1, sbox_logits_2]), dim=-1).sum(dim=0)
        return sbox_logits

class ASCADv2Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.soft_xor = SoftXOR()
        self.soft_gf256mult = SoftGF256Mult()
        self.soft_gf256inv = SoftGF256Inv()
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        batch_size, label_count, class_count = logits.shape
        assert label_count == 18
        assert class_count == 256
        alpha_logits = logits[:, 0:1, :]
        beta_logits = logits[:, 1:2, :]
        masked_sbox_logits = logits[:, 2:18, :]
        logits = self.soft_gf256mult(self.soft_gf256inv(alpha_logits), self.soft_xor(masked_sbox_logits, beta_logits))
        return logits