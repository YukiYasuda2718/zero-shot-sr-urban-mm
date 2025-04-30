import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn


class PointwiseRegressor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        n_hidden: int,
        out_dim: int,
        #
        pos_fc: bool,
        pos_dim: int,
        #
        num_layers: int,
        dropout: float,
        activation: Literal["silu", "relu"],
        return_latent: bool,
    ):
        super().__init__()

        if activation == "silu":
            activ = nn.SiLU()
        elif activation == "relu":
            activ = nn.ReLU()
        else:
            raise Exception()

        self.pos_fc = pos_fc
        if self.pos_fc:
            self.fc = nn.Linear(in_dim + pos_dim, n_hidden)
        else:
            self.fc = nn.Linear(in_dim, n_hidden)

        self.ff = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(n_hidden, n_hidden),
                    activ,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(n_hidden, out_dim)
        self.return_latent = return_latent

    def forward(self, x: torch.Tensor, pos: torch.Tensor = None) -> torch.Tensor:
        if self.pos_fc:
            x = torch.cat([x, pos], dim=-1)
        x = self.fc(x)

        for layer in self.ff:
            x = layer(x)
            x = self.dropout(x)

        x = self.out(x)

        if self.return_latent:
            return x, None
        else:
            return x