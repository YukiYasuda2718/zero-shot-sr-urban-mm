import copy
import dataclasses
from typing import Literal

import torch
from src.models.neural_nets.base_config import BaseModelConfig
from src.models.neural_nets.blocks.regressor import PointwiseRegressor
from src.models.neural_nets.blocks.transformer import TransformerEncoderLayer
from torch import nn


@dataclasses.dataclass()
class ConfigSRNeuralOperatorV01(BaseModelConfig):
    in_channels: int
    is_global_skip_conn: bool
    #
    # Decoder features
    #
    n_decoder_layers: int
    decoder_dropout: float
    decoder_activation: Literal["silu", "relu"]
    #
    # Encoder features
    #
    n_encoder_blocks: int
    d_model: int
    n_head: int
    dim_feedforward: int
    #
    attention_type: Literal["fourier", "galerkin"]
    is_softmax_applied: bool
    xavier_init: float
    diagonal_weight: float
    symmetric_init: bool
    #
    norm_type: Literal["layer", "instance"]
    attn_norm: bool
    norm_eps: float
    #
    dropout: float
    ffn_dropout: float
    #
    pos_dim: int
    encoder_norm: bool
    batch_norm: bool
    activation_type: Literal["relu", "silu", "gelu"]
    use_pos_emb: bool


class SRNeuralOperatorV01(nn.Module):
    def __init__(self, config: ConfigSRNeuralOperatorV01):
        self.config = copy.deepcopy(config)
        super().__init__()

        self.feat_extractor = nn.Conv2d(
            in_channels=self.config.in_channels,
            out_channels=self.config.d_model,
            kernel_size=1,
            bias=True,
        )
        # kernel size is 1, so this Conv is pointwise.

        self.encoder_blocks = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    **self.config.__dict__, return_attn_weight=False
                )
                for _ in range(self.config.n_encoder_blocks)
            ]
        )

        self.dropout = nn.Dropout(self.config.dropout)

        self.regressor = PointwiseRegressor(
            in_dim=self.config.d_model,
            n_hidden=self.config.d_model,
            out_dim=1,
            pos_fc=True,
            pos_dim=self.config.pos_dim,
            num_layers=self.config.n_decoder_layers,
            dropout=self.config.decoder_dropout,
            activation=self.config.decoder_activation,
            return_latent=False,
        )

    def forward(
        self, x: torch.Tensor, pos: torch.Tensor, weight: torch.Tensor = None, **kwargs
    ) -> torch.Tensor:
        #
        assert x.shape[-2:] == pos.shape[-2:]  # x and y dims are the same
        assert x.shape[1] == self.config.in_channels
        assert pos.shape[1] == self.config.pos_dim

        x = self.feat_extractor(x)
        # dims: batch, channel (=d_model), x, y

        batch, channel, size_x, size_y = x.shape
        x = x.view(batch, channel, -1).transpose(1, 2).contiguous()
        # dims: batch, sequence (= x * y), features (=d_model)

        if self.config.is_global_skip_conn:
            x0 = x

        pos = pos.view(batch, self.config.pos_dim, -1).transpose(1, 2).contiguous()

        for encoder in self.encoder_blocks:
            x = encoder(x=x, pos=pos, weight=weight)

        if self.config.is_global_skip_conn:
            x = x + x0

        x = self.dropout(x)
        x = self.regressor(x=x, pos=pos)
        # dims: batch, sequence (= x * y), features (= out_channels = 1)

        batch, _, out = x.shape
        x = x.transpose(1, 2).contiguous().view(batch, out, size_x, size_y)

        return x
