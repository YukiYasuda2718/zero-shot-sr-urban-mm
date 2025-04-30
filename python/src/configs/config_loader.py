import os

from src.configs.base_config import BaseExperimentConfig
from src.configs.config_point_cloud import ConfigPointCloudExperimentNO
from src.configs.config_unseen_data import (
    ConfigUnseenDataExperimentNO,
    ConfigUnseenDataExperimentSRCNN,
)


def load_config(experiment_name: str, config_path: str) -> BaseExperimentConfig:
    if experiment_name == "ZeroShotUnseenData":
        if os.path.basename(config_path).startswith("cnn_"):
            return ConfigUnseenDataExperimentSRCNN.load(config_path)
        elif os.path.basename(config_path).startswith("tno_"):
            return ConfigUnseenDataExperimentNO.load(config_path)
        else:
            raise ValueError()
    elif experiment_name == "ZeroShotPointCloudData":
        return ConfigPointCloudExperimentNO.load(config_path)
    else:
        raise Exception()
