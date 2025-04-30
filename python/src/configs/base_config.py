import dataclasses

from src.data.dataloader import ConfigDataloader
from src.data.datasets.base_config import BaseConfigDataset
from src.models.neural_nets.base_config import BaseModelConfig
from src.utils.base_config import YamlConfig


@dataclasses.dataclass()
class BaseTrainConfig(YamlConfig):
    epochs: int
    early_stopping_patience: int
    loss_name: str
    seed: int
    learning_rate: float


@dataclasses.dataclass()
class BaseExperimentConfig(YamlConfig):
    data: BaseConfigDataset
    loader: ConfigDataloader
    model: BaseModelConfig
    train: BaseTrainConfig
