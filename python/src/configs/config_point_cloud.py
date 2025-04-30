import dataclasses

from src.configs.base_config import BaseExperimentConfig
from src.data.dataloader import ConfigDataloader
from src.data.datasets.dataset_tuv_point_cloud import ConfigDatasetPointCloudTUVWithPos
from src.models.neural_nets.sr_neural_operator_v02 import ConfigSRNeuralOperatorV02


@dataclasses.dataclass()
class ConfigPointCloudExperimentNO(BaseExperimentConfig):
    data: ConfigDatasetPointCloudTUVWithPos
    loader: ConfigDataloader
    model: ConfigSRNeuralOperatorV02
