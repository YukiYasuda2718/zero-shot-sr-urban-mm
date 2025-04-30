import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_


def attention_fourier_type(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_softmax_applied: bool,
    dropout: nn.Dropout,
) -> tuple[torch.Tensor, torch.Tensor]:
    #
    d_k = query.shape[-1]  # dim: batch, ..., sequence_length, feature
    scores = torch.einsum("...ik,...jk->...ij", query, key) / math.sqrt(d_k)

    if is_softmax_applied:
        p_attn = F.softmax(scores, dim=-1)
    else:
        key_len = scores.shape[-1]
        p_attn = scores / key_len

    if dropout is not None:
        p_attn = dropout(p_attn)

    out = torch.einsum("...ij,...jk->...ik", p_attn, value)

    return out, p_attn


def attention_galerkin_type(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_softmax_applied: bool,
    dropout: nn.Dropout,
) -> tuple[torch.Tensor, torch.Tensor]:
    #
    if is_softmax_applied:
        query = query.softmax(dim=-1)
        key = key.softmax(dim=-2)

    scores = torch.einsum("...ij,...ik->...jk", key, value)
    seq_len = query.shape[-2]  # dim: batch, ..., sequence_length, feature
    p_attn = scores / seq_len

    if dropout is not None:
        p_attn = dropout(p_attn)

    out = torch.einsum("...ij,...jk->...ik", query, p_attn)

    return out, p_attn


class Attention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        attention_type: Literal["fourier", "galerkin"],
        #
        is_softmax_applied: bool,
        #
        xavier_init: float,
        diagonal_weight: float,
        symmetric_init: bool,
        #
        norm_type: Literal["layer", "instance"],
        add_norm: bool,
        eps: float,
        #
        pos_dim: int,
        dropout: float,
    ):
        super().__init__()
        assert d_model % n_head == 0
        assert attention_type in ["fourier", "galerkin"]
        assert norm_type in ["layer", "instance"]
        assert xavier_init >= 0.0
        assert diagonal_weight >= 0.0
        assert eps >= 0.0
        assert dropout >= 0.0
        assert isinstance(pos_dim, int) and pos_dim >= 0

        self.attention_type = attention_type
        self.is_softmax_applied = is_softmax_applied

        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head

        # for linear transform of query, key, and value
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.xavier_init = xavier_init
        self.diagonal_weight = diagonal_weight
        self.symmetric_init = symmetric_init

        if self.xavier_init > 0.0:
            self._reset_linear_parameters()

        self.norm_type = norm_type
        self.is_instance_norm = False
        if self.norm_type == "instance":
            self.is_instance_norm = True

        self.add_norm = add_norm
        self.eps = eps
        if self.add_norm:
            self._get_norm()

        self.pos_dim = pos_dim
        if self.pos_dim > 0:
            self.fc = nn.Linear(self.d_model + self.n_head * self.pos_dim, self.d_model)

        self.dropout = nn.Dropout(dropout)
        self.attn_weight = None

    def _reset_linear_parameters(self):
        for param in self.linears.parameters():
            if param.ndim > 1:
                xavier_uniform_(param, gain=self.xavier_init)
                if self.diagonal_weight > 0.0:
                    param.data += self.diagonal_weight * torch.diag(
                        torch.ones(param.size(-1), dtype=torch.float32)
                    )
                if self.symmetric_init:
                    param.data += param.data.T
                    param.data /= 2.0
            else:
                constant_(param, 0)

    def _get_norm(self):
        if self.attention_type == "galerkin":
            if self.norm_type == "instance":
                self.norm_K = self._get_instancenorm()
                self.norm_V = self._get_instancenorm()
                assert self.is_instance_norm
            elif self.norm_type == "layer":
                self.norm_K = self._get_layernorm()
                self.norm_V = self._get_layernorm()
                assert not self.is_instance_norm
            else:
                raise Exception()
        elif self.attention_type == "fourier":
            if self.norm_type == "instance":
                self.norm_K = self._get_instancenorm()
                self.norm_Q = self._get_instancenorm()
                assert self.is_instance_norm
            elif self.norm_type == "layer":
                self.norm_K = self._get_layernorm()
                self.norm_Q = self._get_layernorm()
                assert not self.is_instance_norm
            else:
                raise Exception()
        else:
            raise Exception()

    def _get_layernorm(self):
        return nn.ModuleList(
            [
                nn.LayerNorm(self.d_k, eps=self.eps, elementwise_affine=True)
                for _ in range(self.n_head)
            ]
        )

    def _get_instancenorm(self):
        return nn.ModuleList(
            [
                nn.InstanceNorm1d(self.d_k, eps=self.eps, affine=True)
                for _ in range(self.n_head)
            ]
        )

    def _normalize_galerkin_type(self, key: torch.Tensor, value: torch.Tensor):
        #
        if self.is_instance_norm:
            key = key.transpose(-2, -1)
            value = value.transpose(-2, -1)
            # batch, n_head, sequence, sub_feature -> batch, n_head, sub_feature, sequence
            # nomalization is performed for the `sequence` dim

        key = torch.stack(
            [self.norm_K[i](key[:, i, ...]) for i in range(self.n_head)], dim=1
        )
        value = torch.stack(
            [self.norm_V[i](value[:, i, ...]) for i in range(self.n_head)], dim=1
        )

        if self.is_instance_norm:
            key = key.transpose(-2, -1)
            value = value.transpose(-2, -1)
            # batch, n_head, sub_feature, sequence -> batch, n_head, sequence, sub_feature

        return key, value

    def _normalize_fourier_type(self, query: torch.Tensor, key: torch.Tensor):
        #
        if self.is_instance_norm:
            key = key.transpose(-2, -1)
            query = query.transpose(-2, -1)
            # batch, n_head, sequence, sub_feature -> batch, n_head, sub_feature, sequence
            # nomalization is performed for the `sequence` dim

        key = torch.stack(
            [self.norm_K[i](key[:, i, ...]) for i in range(self.n_head)], dim=1
        )
        query = torch.stack(
            [self.norm_Q[i](query[:, i, ...]) for i in range(self.n_head)], dim=1
        )

        if self.is_instance_norm:
            key = key.transpose(-2, -1)
            query = query.transpose(-2, -1)
            # batch, n_head, sub_feature, sequence -> batch, n_head, sequence, sub_feature

        return query, key

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        pos: torch.Tensor = None,
        weight: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        #
        batch = query.shape[0]

        if weight is not None:
            query, key = weight * query, weight * key

        # batch, sequence, all_feature
        # => batch, sequence, n_head, sub_feature
        # => batch, n_head, sequence, sub_feature
        query, key, value = [
            layer(x).view(batch, -1, self.n_head, self.d_k).transpose(1, 2)
            for layer, x in zip(self.linears, [query, key, value])
        ]

        if self.add_norm:
            if self.attention_type == "fourier":
                query, key = self._normalize_fourier_type(query=query, key=key)
            elif self.attention_type == "galerkin":
                key, value = self._normalize_galerkin_type(key=key, value=value)
            else:
                raise Exception()

        if pos is not None and self.pos_dim > 0:
            assert pos.size(-1) == self.pos_dim
            pos = pos.unsqueeze(1)  # add `head` dim
            pos = pos.repeat([1, self.n_head, 1, 1])
            query, key, value = [
                torch.cat([pos, x], dim=-1) for x in (query, key, value)
            ]

        if self.attention_type == "galerkin":
            x, self.attn_weight = attention_galerkin_type(
                query=query,
                key=key,
                value=value,
                is_softmax_applied=self.is_softmax_applied,
                dropout=self.dropout,
            )
        elif self.attention_type == "fourier":
            x, self.attn_weight = attention_fourier_type(
                query=query,
                key=key,
                value=value,
                is_softmax_applied=self.is_softmax_applied,
                dropout=self.dropout,
            )
        else:
            raise Exception()

        # all dim = batch, n_head, sequence, sub_feature
        out_dim = x.shape[1] * x.shape[-1]  # = n_head x sub_feature

        att_output = x.transpose(1, 2).contiguous().view(batch, -1, out_dim)
        # batch, n_head, sequence, sub_feature
        # =>  batch, sequence, n_head, sub_feature
        # ->  batch, sequence, all_features (= out_dim)

        if pos is not None and self.pos_dim > 0:
            att_output = self.fc(att_output)
            # out_dim is reverted to d_model (= self.n_head * self.d_k)

        return att_output, self.attn_weight
