import sys
import typing
from logging import getLogger

import numpy as np
import torch
from src.models.ssim import SSIM
from torch import nn
from torch.utils.data import DataLoader

if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


logger = getLogger()


def evaluate(
    *,
    dataloader: DataLoader,
    model: nn.Module,
    loss_fns: typing.Dict[str, typing.Callable],
    device: str,
    hide_progress_bar: bool = False,
    num_evaluation_loops: int = 1,
) -> dict[str, torch.Tensor]:
    #
    _ = model.eval()
    dict_loss = {k: [] for k in loss_fns.keys()}
    tot = num_evaluation_loops * len(dataloader)

    with torch.no_grad(), tqdm(total=tot, disable=hide_progress_bar) as t:
        for n in range(num_evaluation_loops):
            t.set_description(f"Loop {n+1}/{num_evaluation_loops}")

            for batch in dataloader:
                for k, v in batch.items():
                    batch[k] = v.to(device)
                preds = model(**batch)

                for loss_name, loss_fn in loss_fns.items():
                    if preds.shape == batch["y"].shape:
                        lss = loss_fn(predicts=preds, targets=batch["y"]).detach().cpu()
                    else:
                        lss = (
                            loss_fn(
                                predicts=preds, targets=batch["y"].view(preds.shape)
                            )
                            .detach()
                            .cpu()
                        )
                    assert lss.ndim == 1  # dims: batch
                    assert lss.shape[0] == batch["y"].shape[0]  # check batch dim
                    dict_loss[loss_name].append(lss)
                t.update(1)

    for k in dict_loss.keys():
        dict_loss[k] = torch.cat(dict_loss[k], dim=0)
        # concat along bach dim

    return dict_loss


class SSIMLoss(nn.Module):
    def __init__(
        self,
        max_value: float,
        scale: float,
        bias: float,
        window_size: int = 11,
        sigma: float = 1.5,
        use_gauss: bool = True,
        offset: float = 0.0,
    ):
        super().__init__()
        self.ssim = SSIM(
            scale=scale,
            bias=bias,
            window_size=window_size,
            sigma=sigma,
            size_average=False,
            max_value=max_value,
            use_gauss=use_gauss,
            offset=offset,
        )

    def forward(self, predicts: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        #
        assert predicts.ndim == targets.ndim == 4  # dims: batch, channel, x, y

        ssims = self.ssim(img1=predicts, img2=targets)

        # mean along channel, x and y
        return 1.0 - torch.mean(ssims, dim=(1, 2, 3))


class L1Loss(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, predicts: torch.Tensor, targets: torch.Tensor):
        #
        assert predicts.ndim == targets.ndim == 4  # dims: batch, channel, x, y

        diffs = torch.abs(predicts - targets)
        return torch.mean(diffs, dim=(1, 2, 3)) * self.scale
