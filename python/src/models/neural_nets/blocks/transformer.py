import math
from typing import Literal

import torch
import torch.nn.functional as F
from src.models.neural_nets.blocks.attention import Attention
from src.models.neural_nets.blocks.position_encoder import SinusoidalPositionalEncoding
from torch import nn


class FeedForward(nn.Module):
    def __init__(
        self,
        in_dim: int,
        dim_feedforward: int,
        out_dim: int,
        batch_norm: bool,
        activation: Literal["relu", "silu", "gelu"],
        dropout: float,
    ):
        assert dropout >= 0.0
        assert activation in ["relu", "silu", "gelu"]
        super().__init__()

        self.lr1 = nn.Linear(in_dim, dim_feedforward)

        if activation == "silu":
            self.activation = nn.SiLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise Exception()

        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(dim_feedforward)

        self.lr2 = nn.Linear(dim_feedforward, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.lr1(x))
        x = self.dropout(x)
        if self.batch_norm:
            # batch, sequence, all_feature -> batch, all_feature, sequence
            # BN is applied to `all_feature` dim
            x = x.permute((0, 2, 1))
            x = self.bn(x)
            x = x.permute((0, 2, 1))  # revert the dim order
        x = self.lr2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        dim_feedforward: int,
        attention_type: Literal["fourier", "galerkin"],
        #
        is_softmax_applied: bool,
        #
        xavier_init: float,
        diagonal_weight: float,
        symmetric_init: bool,
        #
        norm_type: Literal["layer", "instance"],
        attn_norm: bool,
        norm_eps: float,
        #
        pos_dim: int,
        dropout: float,
        ffn_dropout: float,
        #
        encoder_norm: bool,
        batch_norm: bool,
        activation_type: Literal["relu", "silu", "gelu"],
        #
        return_attn_weight: bool,
        use_pos_emb: bool,
        **kwargs
    ):
        super().__init__()

        self.attn = Attention(
            d_model=d_model,
            n_head=n_head,
            attention_type=attention_type,
            is_softmax_applied=is_softmax_applied,
            xavier_init=xavier_init,
            diagonal_weight=diagonal_weight,
            symmetric_init=symmetric_init,
            norm_type=norm_type,
            add_norm=attn_norm,
            eps=norm_eps,
            pos_dim=pos_dim,
            dropout=dropout,
        )

        self.d_model = d_model
        self.n_head = n_head
        self.pos_dim = pos_dim

        self.encoder_norm = encoder_norm
        if self.encoder_norm:
            self.layer_norm1 = nn.LayerNorm(d_model, eps=norm_eps)
            self.layer_norm2 = nn.LayerNorm(d_model, eps=norm_eps)

        self.ff = FeedForward(
            in_dim=d_model,
            out_dim=d_model,
            dim_feedforward=dim_feedforward,
            batch_norm=batch_norm,
            activation=activation_type,
            dropout=ffn_dropout,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.use_pos_emb = use_pos_emb
        if self.use_pos_emb:
            self.pos_emb = SinusoidalPositionalEncoding(
                d_model=d_model, dropout=dropout, max_len=2**13
            )

        self.return_attn_weight = return_attn_weight

    def forward(
        self, x: torch.Tensor, pos: torch.Tensor = None, weight: torch.Tensor = None
    ) -> torch.Tensor:
        #
        # input x dim: batch, sequence, all_feature (= d_model)
        #
        if self.use_pos_emb:
            x = self.pos_emb(x)

        if pos is not None and self.pos_dim > 0:
            att_output, attn_weight = self.attn(
                query=x, key=x, value=x, pos=pos, weight=weight
            )
        else:
            att_output, attn_weight = self.attn(query=x, key=x, value=x, weight=weight)

        x = x + self.dropout1(att_output)

        if self.encoder_norm:
            x = self.layer_norm1(x)

        y = self.ff(x)
        x = x + self.dropout2(y)

        if self.encoder_norm:
            x = self.layer_norm2(x)

        if self.return_attn_weight:
            return x, attn_weight
        else:
            return x
