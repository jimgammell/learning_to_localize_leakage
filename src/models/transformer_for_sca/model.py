from typing import Optional, List, Tuple

import torch
from torch import nn

from .config import TransformerConfig
from .building_blocks import *

class TransformerLayer(nn.Module):
    def __init__(self, config: TransformerConfig, layer_num: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_num = layer_num
        self.pre_attn_norm = NormLayer(self.config, self.layer_num)
        self.attn = AttentionLayer(self.config)
        self.pre_fnn_norm = NormLayer(self.config, self.layer_num)
        self.fnn = FeedForwardLayer(self.config)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.pre_attn_norm(x), mask)
        x = x + self.fnn(self.pre_fnn_norm(x))
        return x

class Head(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.pre_attn_norm = NormLayer(self.config, self.config.layer_count+1)
        self.attn = AttentionPoolingLayer(self.config, output_dim=self.config.output_head_classes)
        if self.config.head_type == 'simple-shared':
            self.to_logits = nn.Identity()
        elif self.config.head_type == 'ascadv1':
            assert False
        else:
            assert False
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.attn(self.pre_attn_norm(x), mask)
        x = self.to_logits(x)
        return x

class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.patchifier = Patchifier(self.config)
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(self.config, layer_num) for layer_num in range(1, self.config.layer_count+1)
        ])
        self.head = Head(self.config) #self.heads = AttentionPoolingLayer(self.config)
        self.input_shape = self.config.input_shape
        self.output_classes = self.config.output_classes
    
    def forward(self, x, noise: Optional[torch.Tensor] = None):
        if self.config.noise_conditional:
            assert noise is not None
            x, mask = self.patchifier(x, noise)
        else:
            assert noise is None
            x, mask = self.patchifier(x)
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, mask)
        x = self.head(x, mask) #self.heads(x)
        return x
    
    def get_params_based_on_should_weight_decay(self) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        yes_decay, no_decay = [], []
        for module in self.modules():
            if isinstance(module, NormLayer):
                no_decay.append(module.weight)
            elif isinstance(module, AttentionLayer):
                yes_decay.append(module.to_qkv.weight)
                yes_decay.append(module.to_out.weight)
                if module.to_qkv.bias is not None:
                    no_decay.append(module.to_qkv.bias)
                if module.to_out.bias is not None:
                    no_decay.append(module.to_out.bias)
            elif isinstance(module, FeedForwardLayer):
                yes_decay.append(module.w12.weight)
                yes_decay.append(module.w3.weight)
                if module.w12.bias is not None:
                    no_decay.append(module.w12.bias)
                if module.w3.bias is not None:
                    no_decay.append(module.w3.bias)
            elif isinstance(module, AttentionPoolingLayer):
                yes_decay.append(module.to_kv.weight)
                no_decay.append(module.pool_queries)
                if module.to_kv.bias is not None:
                    no_decay.append(module.to_kv.bias)
                if self.config.shared_head:
                    yes_decay.append(module.to_out.weight)
                    if module.to_out.bias is not None:
                        no_decay.append(module.to_out.bias)
                else:
                    for head in module.to_out:
                        yes_decay.append(head.weight)
                        if head.bias is not None:
                            no_decay.append(head.bias)
            elif isinstance(module, Patchifier):
                no_decay.append(module.patch_embedding.weight)
                if module.patch_embedding.bias is not None:
                    no_decay.append(module.patch_embedding.bias)
        assert all(param is not None for param in yes_decay)
        assert all(param is not None for param in no_decay)
        return yes_decay, no_decay