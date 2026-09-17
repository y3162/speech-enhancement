from functools import partial

import torch
import torch.nn as nn
from mamba_ssm.models.mixer_seq_simple import _init_weights
from mamba_ssm.modules.block import Block
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.ops.triton.layer_norm import RMSNorm


def create_block(
    d_model: int,
    d_state: int,
    d_conv: int,
    expand: int,
    norm_epsilon: float,
    layer_idx: int = 0,
) -> Block:
    mixer_cls = partial(
        Mamba,
        layer_idx=layer_idx,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
    )
    norm_cls = partial(RMSNorm, eps=norm_epsilon)
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls=nn.Identity,
        norm_cls=norm_cls,
        fused_add_norm=False,
        residual_in_fp32=False,
    )
    block.layer_idx = layer_idx
    return block


class MambaBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        d_state: int,
        d_conv: int,
        expand: int,
        norm_epsilon: float,
    ) -> None:
        super().__init__()
        n_layer = 1
        self.forward_blocks = nn.ModuleList(
            create_block(in_channels, d_state, d_conv, expand, norm_epsilon)
            for _ in range(n_layer)
        )
        self.backward_blocks = nn.ModuleList(
            create_block(in_channels, d_state, d_conv, expand, norm_epsilon)
            for _ in range(n_layer)
        )
        self.apply(partial(_init_weights, n_layer=n_layer))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_forward, x_backward = x.clone(), torch.flip(x, [1])
        residual_forward, residual_backward = None, None
        for layer in self.forward_blocks:
            x_forward, residual_forward = layer(x_forward, residual_forward)
        y_forward = (
            x_forward + residual_forward if residual_forward is not None else x_forward
        )
        for layer in self.backward_blocks:
            x_backward, residual_backward = layer(x_backward, residual_backward)
        y_backward = (
            torch.flip(x_backward + residual_backward, [1])
            if residual_backward is not None
            else torch.flip(x_backward, [1])
        )
        return torch.cat([y_forward, y_backward], -1)


def mamba_block_from_cfg(in_channels: int, cfg) -> MambaBlock:
    return MambaBlock(
        in_channels=in_channels,
        d_state=cfg.d_state,
        d_conv=cfg.d_conv,
        expand=cfg.expand,
        norm_epsilon=cfg.norm_epsilon,
    )
