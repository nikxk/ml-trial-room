import torch
import torch.nn as nn

from typing import Optional, Tuple, List, Callable
from perceiver.model.core import InputAdapter, checkpoint_wrapper
from perceiver.model.core.modules import CrossAttentionLayer, SelfAttentionBlock
from perceiver.model.core.utils import init_parameters



class PerceiverEncoder(nn.Module):
    def __init__(
        self,
        input_adapter: InputAdapter,
        num_latents: int,
        num_latent_channels: int,
        num_cross_attention_heads: int = 4,
        num_cross_attention_qk_channels: Optional[int] = None,
        num_cross_attention_v_channels: Optional[int] = None,
        num_cross_attention_layers: int = 1,
        first_cross_attention_layer_shared: bool = False,
        cross_attention_widening_factor: int = 1,
        num_self_attention_heads: int = 4,
        num_self_attention_qk_channels: Optional[int] = None,
        num_self_attention_v_channels: Optional[int] = None,
        num_self_attention_layers_per_block: int = 6,
        num_self_attention_blocks: int = 1,
        first_self_attention_block_shared: bool = True,
        self_attention_widening_factor: int = 1,
        dropout: float = 0.0,
        init_scale: float = 0.02,
        activation_checkpointing: bool = False,
        activation_offloading: bool = False,
    ):
        """
        for the most part, source: https://github.com/krasserm/perceiver-io/blob/main/perceiver/model/core/modules.py

        changes: 
        - latent has to be passed to the encoder
        """
        super().__init__()

        # self.latent_provider = TrainableQueryProvider(num_latents, num_latent_channels, init_scale=init_scale)
        self.input_adapter = input_adapter

        if num_cross_attention_layers <= 0:
            raise ValueError("num_cross_attention_layers must be > 0")

        if num_self_attention_blocks <= 0:
            raise ValueError("num_self_attention_blocks must be > 0")

        if num_cross_attention_layers > num_self_attention_blocks:
            raise ValueError("num_cross_attention_layers must be <= num_self_attention_blocks")

        self.num_cross_attention_layers = num_cross_attention_layers
        self.num_self_attention_blocks = num_self_attention_blocks

        self.first_cross_attention_layer_shared = first_cross_attention_layer_shared
        self.first_self_attention_block_shared = first_self_attention_block_shared

        def cross_attn():
            layer = CrossAttentionLayer(
                num_heads=num_cross_attention_heads,
                num_q_input_channels=num_latent_channels,
                num_kv_input_channels=input_adapter.num_input_channels,
                num_qk_channels=num_cross_attention_qk_channels,
                num_v_channels=num_cross_attention_v_channels,
                widening_factor=cross_attention_widening_factor,
                dropout=dropout,
            )
            return (
                checkpoint_wrapper(layer, offload_to_cpu=activation_offloading) if activation_checkpointing else layer
            )

        def self_attn():
            return SelfAttentionBlock(
                num_layers=num_self_attention_layers_per_block,
                num_heads=num_self_attention_heads,
                num_channels=num_latent_channels,
                num_qk_channels=num_self_attention_qk_channels,
                num_v_channels=num_self_attention_v_channels,
                widening_factor=self_attention_widening_factor,
                dropout=dropout,
                activation_checkpointing=activation_checkpointing,
                activation_offloading=activation_offloading,
            )

        self.cross_attn_1 = cross_attn()
        self.self_attn_1 = self_attn()

        if self.extra_cross_attention_layer:
            self.cross_attn_n = cross_attn()

        if self.extra_self_attention_block:
            self.self_attn_n = self_attn()

        self._init_parameters(init_scale)

    def _init_parameters(self, init_scale: float):
        with torch.no_grad():
            init_parameters(self, init_scale)

    @property
    def extra_cross_attention_layer(self):
        return self.num_cross_attention_layers > 1 and not self.first_cross_attention_layer_shared

    @property
    def extra_self_attention_block(self):
        return self.num_self_attention_blocks > 1 and not self.first_self_attention_block_shared

    def forward(self, x, latent, pad_mask=None, return_adapted_input=False):
        b, *_ = x.shape

        x_adapted = self.input_adapter(x)
        x_latent = latent

        x_latent = self.cross_attn_1(x_latent, x_adapted, pad_mask=pad_mask)
        x_latent = self.self_attn_1(x_latent)

        cross_attn_n = self.cross_attn_n if self.extra_cross_attention_layer else self.cross_attn_1
        self_attn_n = self.self_attn_n if self.extra_self_attention_block else self.self_attn_1

        for i in range(1, self.num_self_attention_blocks):
            if i < self.num_cross_attention_layers:
                x_latent = cross_attn_n(x_latent, x_adapted, pad_mask=pad_mask)
            x_latent = self_attn_n(x_latent)

        if return_adapted_input:
            return x_latent, x_adapted
        else:
            return x_latent