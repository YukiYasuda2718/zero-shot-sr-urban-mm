import random
import sys
import typing
from logging import getLogger

import numpy as np
import torch
import torch.distributed as dist
from src.utils.average_meter import AverageMeter
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Sampler

logger = getLogger()


def optimize(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.functional,
    optimizer: Optimizer,
    device: str,
    epoch: int,
    mode: typing.Literal["train", "valid", "test"],
) -> float:
    #
    loss_meter = AverageMeter()

    if mode == "train":
        model.train()
    elif mode in ["valid", "test"]:
        model.eval()
    else:
        raise NotImplementedError(f"{mode} is not supported.")

    random.seed(epoch)
    np.random.seed(epoch)

    for batch in dataloader:
        for k, v in batch.items():
            batch[k] = v.to(device)

        if mode == "train":
            preds = model(**batch)
            loss = loss_fn(predicts=preds, targets=batch["y"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                preds = model(**batch)
                loss = loss_fn(predicts=preds, targets=batch["y"])

        loss_meter.update(loss.item(), n=batch["x"].shape[0])

    logger.info(f"{mode} error: avg loss = {loss_meter.avg:.8f}")

    return loss_meter.avg


def optimize_ddp(
    dataloader: DataLoader,
    sampler: Sampler,
    model: nn.Module,
    loss_fn: nn.functional,
    optimizer: Optimizer,
    epoch: int,
    rank: int,
    world_size: int,
    mode: typing.Literal["train", "valid", "test"],
) -> float:
    #
    mean_loss, cnt = 0.0, 0

    if mode == "train":
        model.train()
    elif mode in ["valid", "test"]:
        model.eval()
    else:
        raise NotImplementedError(f"{mode} is not supported.")

    sampler.set_epoch(epoch)
    random.seed(epoch)
    np.random.seed(epoch)

    for batch in dataloader:
        for k, v in batch.items():
            batch[k] = v.to(rank)

        if mode == "train":
            preds = model(**batch)
            loss = loss_fn(predicts=preds, targets=batch["y"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                preds = model(**batch)
                loss = loss_fn(predicts=preds, targets=batch["y"])

        mean_loss += loss * batch["x"].shape[0]
        cnt += batch["x"].shape[0]
    mean_loss /= cnt

    dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)

    return mean_loss.item() / world_size
