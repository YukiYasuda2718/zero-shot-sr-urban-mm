import copy
from logging import INFO, WARNING, getLogger
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from src.configs.base_config import BaseExperimentConfig
from src.configs.config_averaged_data import ConfigAveragedDataExperimentNO
from src.configs.config_point_cloud import ConfigPointCloudExperimentNO
from src.configs.config_unseen_data import (
    ConfigUnseenDataExperimentNO,
    ConfigUnseenDataExperimentSRCNN,
)
from src.data.dataloader import make_dataloaders_and_samplers
from src.models.model_maker import make_model
from torch.utils.data import DataLoader

logger = getLogger()


def load_model(config: BaseExperimentConfig, weight_path: str, device: str):
    logger.setLevel(WARNING)
    model = make_model(config.model).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    _ = model.eval()
    logger.setLevel(INFO)
    return model


def get_test_dataloader_05m(
    config: Union[
        ConfigUnseenDataExperimentNO,
        ConfigUnseenDataExperimentSRCNN,
        ConfigAveragedDataExperimentNO,
        ConfigPointCloudExperimentNO,
    ],
    root_dir: str,
    batch: int = 4,
    size: int = 160,
) -> DataLoader:
    #
    _config = copy.deepcopy(config)
    _config.data.hr_name = "tokyo_05m"
    _config.data.x_full_size = 320
    _config.data.y_full_size = 320

    if isinstance(_config, ConfigUnseenDataExperimentNO) or isinstance(
        _config, ConfigUnseenDataExperimentSRCNN
    ):
        _config.data.x_segmented_size = size
        _config.data.y_segmented_size = size
        _config.data.x_crop_size = size
        _config.data.y_crop_size = size

    elif isinstance(_config, ConfigAveragedDataExperimentNO):
        _config.data.x_crop_size = size
        _config.data.y_crop_size = size

    elif isinstance(_config, ConfigPointCloudExperimentNO):
        pass

    else:
        raise Exception()

    _config.loader.batch_size = batch

    logger.setLevel(WARNING)
    dataloaders, _ = make_dataloaders_and_samplers(
        root_dir=root_dir,
        config_loader=_config.loader,
        config_data=_config.data,
        train_valid_test_kinds=["test"],
    )
    logger.setLevel(INFO)

    return dataloaders["test"]


def get_test_dataloader_20m(
    config: Union[
        ConfigUnseenDataExperimentNO,
        ConfigUnseenDataExperimentSRCNN,
        ConfigAveragedDataExperimentNO,
        ConfigPointCloudExperimentNO,
    ],
    root_dir: str,
    batch: int = 64,
    size: int = 40,
) -> DataLoader:
    #
    _config = copy.deepcopy(config)
    _config.data.hr_name = "tokyo_20m_vz_no_s2srad"
    _config.data.x_full_size = 80
    _config.data.y_full_size = 80

    if isinstance(_config, ConfigUnseenDataExperimentNO) or isinstance(
        _config, ConfigUnseenDataExperimentSRCNN
    ):
        _config.data.x_segmented_size = size
        _config.data.y_segmented_size = size
        _config.data.x_crop_size = size
        _config.data.y_crop_size = size

    elif isinstance(_config, ConfigAveragedDataExperimentNO):
        _config.data.x_crop_size = size
        _config.data.y_crop_size = size

    elif isinstance(_config, ConfigPointCloudExperimentNO):
        pass

    else:
        raise Exception()

    _config.loader.batch_size = batch

    logger.setLevel(WARNING)
    dataloaders, _ = make_dataloaders_and_samplers(
        root_dir=root_dir,
        config_loader=_config.loader,
        config_data=_config.data,
        train_valid_test_kinds=["test"],
    )
    logger.setLevel(INFO)

    return dataloaders["test"]


def set_config_for_unseen_test(
    config: Union[ConfigUnseenDataExperimentNO, ConfigUnseenDataExperimentSRCNN],
    config_name: str,
):
    #
    assert len(config.data.used_segment_indices) == 3

    _config = copy.deepcopy(config)

    target_idx = None
    for i in [0, 1, 2, 3]:
        if i not in config.data.used_segment_indices:
            target_idx = i
            _config.data.used_segment_indices = [target_idx]
            break

    assert f"_i{target_idx}_" in config_name, f"{config_name=}, {target_idx=}"
    assert len(_config.data.used_segment_indices) == 1
    assert _config.data.used_segment_indices[0] == target_idx

    return _config, target_idx
