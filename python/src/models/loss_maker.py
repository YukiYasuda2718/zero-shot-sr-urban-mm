import copy
from logging import getLogger

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

logger = getLogger()


def make_loss(loss_name: str) -> nn.Module:
    if loss_name == "L2":
        logger.info("L2 loss is created.")
        return L2Loss()
    elif loss_name == "L1":
        logger.info("L1 loss is created.")
        return L1Loss()
    else:
        raise NotImplementedError(f"{loss_name} is not supported.")


class L2Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, predicts: torch.Tensor, targets: torch.Tensor):
        return self.loss(predicts, targets)


class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, predicts: torch.Tensor, targets: torch.Tensor):
        return self.loss(predicts, targets)


