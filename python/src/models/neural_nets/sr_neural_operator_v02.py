import copy
import dataclasses
from typing import Literal

import torch
from src.models.neural_nets.base_config import BaseModelConfig
from src.models.neural_nets.blocks.regressor import PointwiseRegressor
from src.models.neural_nets.blocks.transformer import TransformerEncoderLayer
from torch import nn


@dataclasses.dataclass()
class ConfigSRNeuralOperatorV02(BaseModelConfig):
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


class SRNeuralOperatorV02(nn.Module):
    def __init__(self, config: ConfigSRNeuralOperatorV02):
        self.config = copy.deepcopy(config)
        super().__init__()

        self.feat_extractor = nn.Linear(
            in_features=self.config.in_channels,
            out_features=self.config.d_model,
            bias=True,
        )

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

        self.reshaped_size = None

    def forward(
        self, x: torch.Tensor, pos: torch.Tensor, weight: torch.Tensor = None, **kwargs
    ) -> torch.Tensor:
        #
        assert x.ndim == pos.ndim == 3  # batch, channel, coord (= sequence)
        assert x.shape[2] == pos.shape[2]
        # dim for coordinate, i.e., num of coordinate points

        assert x.shape[1] == self.config.in_channels
        assert pos.shape[1] == self.config.pos_dim

        x = self.feat_extractor(x.permute(0, 2, 1)).permute(0, 2, 1)
        # dims: batch, channel (=d_model), coord (= sequence)

        batch, channel, coord = x.shape
        x = x.transpose(1, 2).contiguous()
        # dims: batch, sequence (= coord), features (=d_model)

        if self.config.is_global_skip_conn:
            x0 = x

        pos = pos.transpose(1, 2).contiguous()

        for encoder in self.encoder_blocks:
            x = encoder(x=x, pos=pos, weight=weight)

        if self.config.is_global_skip_conn:
            x = x + x0

        x = self.dropout(x)
        x = self.regressor(x=x, pos=pos)
        # dims: batch, sequence (= coord), features (= out_channels = 1)

        x = x.transpose(1, 2).contiguous()

        if self.reshaped_size is None:
            return x
        else:
            batch, chan, _ = x.shape
            return x.view((batch, chan) + self.reshaped_size)
