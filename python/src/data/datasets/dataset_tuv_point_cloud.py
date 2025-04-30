import copy
import dataclasses
import datetime
import glob
import os
from logging import getLogger
from typing import Literal, Union

import numpy as np
import torch
import torch.nn.functional as F
from src.data.datasets.base_config import BaseConfigDataset
from torch.utils.data import Dataset

logger = getLogger()


@dataclasses.dataclass()
class ConfigDatasetPointCloudTUVWithPos(BaseConfigDataset):
    lr_name: str
    hr_name: str
    dtype: Literal["float32", "float64"]
    target_variables: list[str]
    input_variables: list[str]
    interpolation_mode: str
    scales: dict[str, float]
    biases: dict[str, float]
    x_full_size: int
    y_full_size: int
    concat_topography_with_position: bool
    discarded_minute_range: list[float]
    use_random_points: bool

    def __post_init__(self):
        assert self.x_full_size > 0
        assert self.y_full_size > 0
        assert len(self.discarded_minute_range) == 2
        assert self.discarded_minute_range[0] < self.discarded_minute_range[1]


class DatasetPointCloudTUVWithPos(Dataset):

    def __init__(self, config: ConfigDatasetPointCloudTUVWithPos, **kwargs):
        self.config = copy.deepcopy(config)
        logger.info(
            f"Input DatasetPointCloudTUVWithPos config = {self.config.to_json_str()}"
        )

        if self.config.dtype == "float32":
            self.dtype = torch.float32
        elif self.config.dtype == "float64":
            self.dtype = torch.float64
        else:
            raise Exception()

        self._set_positions()
        self._set_all_file_pairs()
        self._set_hr_bldg_and_hr_landuse()
        self._get_random_points()

    def _set_positions(self):
        xs = torch.linspace(-1, 1, self.config.x_full_size, dtype=self.dtype)
        ys = torch.linspace(-1, 1, self.config.y_full_size, dtype=self.dtype)
        self.xs, self.ys = torch.meshgrid(xs, ys, indexing="ij")

    def _set_all_file_pairs(self):
        hr_files, lr_files = [], []

        for dir_path in self.config.dir_paths:
            # datetime format is ISO8601, e.g., 2023040500T130000
            for file_path in glob.glob(f"{dir_path}/20*T*.npz"):
                if file_path.endswith(f"{self.config.hr_name}.npz"):

                    # e.g., basename = 20130709T040100_tokyo_20m_vz_no_s2srad.npz
                    dt = datetime.datetime.strptime(
                        os.path.basename(file_path).split("_")[0], "%Y%m%dT%H%M%S"
                    )

                    if (
                        self.config.discarded_minute_range[0]
                        <= dt.minute
                        <= self.config.discarded_minute_range[1]
                    ):
                        continue

                    hr_files.append(file_path)

                    lr_path = file_path.replace(
                        self.config.hr_name, self.config.lr_name
                    )
                    assert os.path.exists(lr_path)
                    lr_files.append(lr_path)

        assert len(hr_files) == len(lr_files)

        self.hr_files = hr_files
        self.lr_files = lr_files

    def _set_hr_bldg_and_hr_landuse(self):
        bldg: Union[None, np.ndarray] = None
        land: Union[None, np.ndarray] = None

        for dir_path in self.config.dir_paths:
            path = f"{dir_path}/bldg_landuse_{self.config.hr_name}.npz"
            data = np.load(path)
            _bldg = data["building_height"]
            _land = data["landuse_index"]

            if bldg is None:
                bldg = _bldg
            else:
                np.testing.assert_array_equal(bldg, _bldg)
                assert bldg.shape == (self.config.x_full_size, self.config.y_full_size)

            if land is None:
                land = _land
            else:
                np.testing.assert_array_equal(land, _land)
                assert land.shape == (self.config.x_full_size, self.config.y_full_size)

        self.bldg = torch.from_numpy(bldg).to(self.dtype)
        self.land = torch.from_numpy(land).to(self.dtype)

    def _get_random_points(self):
        n_points = len(self.hr_files)
        s = (n_points, self.config.x_full_size * self.config.y_full_size)

        torch.manual_seed(42)
        self.random_Xs = torch.rand(s, dtype=self.dtype) * 2 - 1
        self.random_Ys = torch.rand(s, dtype=self.dtype) * 2 - 1

        # F.grid_sample requires the input to be in the range of [-1, 1]
        assert torch.max(self.random_Xs).item() <= 1.0
        assert torch.min(self.random_Xs).item() >= -1.0
        assert torch.max(self.random_Ys).item() <= 1.0
        assert torch.min(self.random_Ys).item() >= -1.0

    def _get_scale(self, var_name: str) -> tuple[float, float]:
        if var_name == "tm02m":
            s, b = self.config.scales["gt"], self.config.biases["gt"]
        elif var_name == "tm_g1":
            logger.debug("tm_g1 is used.")
            s, b = self.config.scales["tm_g1"], self.config.biases["tm_g1"]
        elif var_name.startswith("tm"):
            s, b = self.config.scales["tm"], self.config.biases["tm"]
        elif var_name.startswith("vl"):
            s, b = self.config.scales["vl"], self.config.biases["vl"]
        elif var_name.startswith("vp"):
            s, b = self.config.scales["vp"], self.config.biases["vp"]
        elif var_name == "bldg":
            s, b = self.config.scales["bldg"], self.config.biases["bldg"]
        elif var_name == "land":
            s, b = self.config.scales["land"], self.config.biases["land"]
        else:
            raise Exception()

        return s, b

    def _scale(self, data: torch.Tensor, var_name: str) -> torch.Tensor:
        s, b = self._get_scale(var_name)
        return (data - b) / s

    def _inv_scale(self, data: torch.Tensor, var_name: str) -> torch.Tensor:
        s, b = self._get_scale(var_name)
        return s * data + b

    def _make_target_data(self, hr: dict[str, np.ndarray]) -> torch.Tensor:
        data = []
        for var_name in self.config.target_variables:
            d = torch.from_numpy(hr[var_name]).to(self.dtype)
            d = self._scale(d, var_name)
            data.append(d)
            logger.debug(f"Target {var_name} shape = {d.shape}")
        return torch.stack(data, dim=0)  # dim: channel, x, y

    def _make_input_data(
        self, lr: dict[str, np.ndarray]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _data = []
        for var_name in self.config.input_variables:
            d = torch.from_numpy(lr[var_name]).to(self.dtype)
            d = self._scale(d, var_name)
            _data.append(d)
            logger.debug(f"Input {var_name} shape = {d.shape}")
        data = torch.stack(_data, dim=0)  # dim: channel, x, y

        data = F.interpolate(
            data[None, ...],
            size=(self.config.x_full_size, self.config.y_full_size),
            mode=self.config.interpolation_mode,
        ).squeeze()

        bldg = self.bldg.detach().clone()[None, ...]
        bldg = self._scale(bldg, "bldg").to(self.dtype)

        land = self.land.detach().clone()[None, ...]
        land = self._scale(land, "land").to(self.dtype)

        xs = self.xs.detach().clone()[None, ...].to(self.dtype)
        ys = self.ys.detach().clone()[None, ...].to(self.dtype)

        if self.config.concat_topography_with_position:
            pos = torch.cat([xs, ys, bldg, land], dim=0)
        else:
            pos = torch.cat([xs, ys], dim=0)

        return pos, torch.cat([data, land, bldg], dim=0)

    def __len__(self) -> int:
        return len(self.hr_files)

    def _interpolate_into_random_points(
        self, data: torch.Tensor, idx: int
    ) -> torch.Tensor:
        #
        assert data.ndim == 3
        n_channels, n_x, n_y = data.shape

        random_xs = self.random_Xs[idx]
        random_ys = self.random_Ys[idx]
        assert random_xs.shape == random_ys.shape == (n_x * n_y,)

        grid = torch.stack([random_xs, random_ys], dim=-1).view(1, n_x, n_y, 2)

        interpolated_data = F.grid_sample(
            data[None, ...],
            grid,
            mode="bicubic",
            align_corners=True,
            padding_mode="border",
        ).squeeze(0)
        assert interpolated_data.shape == (n_channels, n_x, n_y)
        logger.debug("Used bicubic interpolation for random points")

        return interpolated_data.view(n_channels, -1)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        #
        hr_path = self.hr_files[idx]
        hr = np.load(hr_path)
        logger.debug(f"hr keys = {list(hr.keys())}")

        lr_path = self.lr_files[idx]
        lr = np.load(lr_path)
        logger.debug(f"lr keys = {list(lr.keys())}")

        target_data = self._make_target_data(hr)
        pos, input_data = self._make_input_data(lr)

        s = (self.config.x_full_size, self.config.y_full_size)
        assert target_data.shape[-2:] == input_data.shape[-2:] == pos.shape[-2:] == s

        full_data = torch.cat([target_data, pos, input_data], dim=0)

        if not self.config.use_random_points:
            logger.debug("No use of random points")
            logger.debug(f"full_data shape = {full_data.shape}")
            stacked = full_data.view(full_data.shape[:-2] + (-1,))
            logger.debug(f"stacked shape = {stacked.shape}")
        else:
            logger.debug("Use of random points")
            logger.debug(f"full_data shape = {full_data.shape}")
            stacked = self._interpolate_into_random_points(full_data, idx)
            logger.debug(f"stacked shape = {stacked.shape}")

        n_pos = 4 if self.config.concat_topography_with_position else 2
        assert n_pos == pos.shape[0]

        interp_target = stacked[0 : target_data.shape[0]]
        interp_pos = stacked[target_data.shape[0] : target_data.shape[0] + n_pos]
        interp_input = stacked[target_data.shape[0] + n_pos :]

        return dict(x=interp_input, pos=interp_pos, y=interp_target)
