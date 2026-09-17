from functools import partial

import torch
import torch.nn as nn
from mamba_ssm.models.mixer_seq_simple import _init_weights
from mamba_ssm.modules.block import Block
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.ops.triton.layer_norm import RMSNorm


def create_block(d_model: int, cfg) -> Block:
    mixer_cls = partial(
        Mamba,
        layer_idx=0,
        d_state=cfg.d_state,
        d_conv=cfg.d_conv,
        expand=cfg.expand,
    )
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls=nn.Identity,
        norm_cls=partial(RMSNorm, eps=cfg.norm_epsilon),
        fused_add_norm=False,
        residual_in_fp32=False,
    )
    block.layer_idx = 0
    return block


class MambaBlock(nn.Module):
    """Bidirectional Mamba. [B, L, C] -> [B, L, 2C] by concatenating forward and backward outputs."""

    def __init__(self, in_channels: int, cfg) -> None:
        super().__init__()
        self.forward_block = create_block(in_channels, cfg)
        self.backward_block = create_block(in_channels, cfg)
        self.apply(partial(_init_weights, n_layer=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_forward, residual = self.forward_block(x.clone(), None)
        y_forward = y_forward + residual
        y_backward, residual = self.backward_block(torch.flip(x, [1]), None)
        y_backward = torch.flip(y_backward + residual, [1])
        return torch.cat([y_forward, y_backward], -1)
