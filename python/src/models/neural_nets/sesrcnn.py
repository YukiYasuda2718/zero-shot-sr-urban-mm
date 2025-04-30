import dataclasses
from logging import getLogger

import torch
import torch.nn.functional as F
from src.models.neural_nets.base_config import BaseModelConfig
from torch import nn

logger = getLogger()


@dataclasses.dataclass()
class ConfigSESRCNN(BaseModelConfig):
    in_channels: int
    is_globally_skipped: bool
    lr_tm_channel: int


class SESRCNN(nn.Module):
    def __init__(self, config: ConfigSESRCNN):
        super().__init__()

        logger.info(f"Input SESRCNN config = {config.to_json_str()}")
        self.is_skipped = config.is_globally_skipped
        self.lr_tm_channel = config.lr_tm_channel

        n_features = 64 * config.in_channels

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=config.in_channels,
                out_channels=n_features,
                kernel_size=9,
                padding=4,
                groups=config.in_channels,
            ),
            nn.ReLU(),
        )

        self.mapper = nn.Sequential(
            nn.Conv2d(in_channels=n_features, out_channels=32, kernel_size=1), nn.ReLU()
        )
        self.reconstructor = nn.Conv2d(
            in_channels=32, out_channels=1, kernel_size=5, padding=2
        )

        self.se_block = nn.Sequential(
            nn.Linear(in_features=n_features, out_features=n_features),
            nn.ReLU(),
            nn.Linear(in_features=n_features, out_features=n_features),
            nn.Sigmoid(),
        )

    def _calc_channel_attention(self, X: torch.Tensor) -> torch.Tensor:
        atts = F.avg_pool2d(X, kernel_size=X.shape[-2:])
        # specify x and y sizes (last two dims)

        atts = atts.view(atts.shape[:2])  # dims = (batch, channel)
        atts = self.se_block(atts)

        atts = atts.view(atts.shape[:2] + (1, 1))  # add x and y dims

        return atts

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        y0 = x[:, self.lr_tm_channel : self.lr_tm_channel + 1]

        feats = self.feature_extractor(x)
        atts = self._calc_channel_attention(feats)
        feats = feats * atts

        y = self.mapper(feats)
        y = self.reconstructor(y)

        if self.is_skipped:
            return y + y0  # global residual connection
        else:
            return y
