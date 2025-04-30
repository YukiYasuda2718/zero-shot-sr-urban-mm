import dataclasses

from src.configs.base_config import BaseExperimentConfig
from src.data.dataloader import ConfigDataloader
from src.data.datasets.dataset_segmented_tuv_with_pos import (
    ConfigDatasetSegmentedTUVWithPos,
)
from src.models.neural_nets.sesrcnn_with_pos import ConfigSESRCNNWithPos
from src.models.neural_nets.sr_neural_operator_v01 import ConfigSRNeuralOperatorV01


@dataclasses.dataclass()
class ConfigUnseenDataExperimentNO(BaseExperimentConfig):
    data: ConfigDatasetSegmentedTUVWithPos
    loader: ConfigDataloader
    model: ConfigSRNeuralOperatorV01


@dataclasses.dataclass()
class ConfigUnseenDataExperimentSRCNN(BaseExperimentConfig):
    data: ConfigDatasetSegmentedTUVWithPos
    loader: ConfigDataloader
    model: ConfigSESRCNNWithPos
