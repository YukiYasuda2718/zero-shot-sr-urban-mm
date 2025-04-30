import copy
import dataclasses
import datetime
import glob
import os
from collections import OrderedDict
from logging import getLogger
from typing import Literal, Union

import numpy as np
import torch
import torch.nn.functional as F
from src.data.datasets.base_config import BaseConfigDataset
from src.utils.random_crop import RandomCrop2D
from torch.utils.data import Dataset

logger = getLogger()


@dataclasses.dataclass()
class ConfigDatasetSegmentedTUVWithPos(BaseConfigDataset):
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
    x_segmented_size: int
    y_segmented_size: int
    x_crop_size: int
    y_crop_size: int
    concat_topography_with_position: bool
    used_segment_indices: list[int]
    discarded_minute_range: list[float]

    def __post_init__(self):
        assert self.x_full_size % self.x_segmented_size == 0
        assert self.y_full_size % self.y_segmented_size == 0
        assert self.x_full_size >= self.x_segmented_size >= self.x_crop_size > 0
        assert self.y_full_size >= self.y_segmented_size >= self.y_crop_size > 0
        assert len(self.discarded_minute_range) == 2
        assert self.discarded_minute_range[0] < self.discarded_minute_range[1]


class DatasetSegmentedTUVWithPos(Dataset):

    def __init__(self, config: ConfigDatasetSegmentedTUVWithPos, **kwargs):
        self.config = copy.deepcopy(config)
        logger.info(
            f"Input DatasetSegmentedTUVWithPos config = {self.config.to_json_str()}"
        )

        if self.config.dtype == "float32":
            self.dtype = torch.float32
        elif self.config.dtype == "float64":
            self.dtype = torch.float64
        else:
            raise Exception()

        self.random_crop = RandomCrop2D(
            img_sz=(self.config.x_segmented_size, self.config.y_segmented_size),
            crop_sz=(self.config.x_crop_size, self.config.y_crop_size),
        )

        self._set_positions()
        self._set_all_file_pairs()
        self._set_hr_bldg_and_hr_landuse()
        self._make_segment_indices()

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
        bldg, land = None, None
        for dir_path in self.config.dir_paths:
            path = f"{dir_path}/bldg_landuse_{self.config.hr_name}.npz"
            data = np.load(path)
            _bldg = data["building_height"]
            _land = data["landuse_index"]

            if bldg is None and land is None:
                bldg = _bldg
                land = _land
            else:
                np.testing.assert_array_equal(bldg, _bldg)
                np.testing.assert_array_equal(land, _land)

        assert (
            bldg.shape
            == land.shape
            == (self.config.x_full_size, self.config.y_full_size)
        )

        self.bldg = torch.from_numpy(bldg).to(self.dtype)
        self.land = torch.from_numpy(land).to(self.dtype)

    def _get_scale(self, var_name: str) -> tuple[float, float]:
        if var_name == "tm02m":
            s, b = self.config.scales["gt"], self.config.biases["gt"]
        elif var_name == "tm_g1":
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

    def _scale(
        self, data: Union[np.ndarray, torch.Tensor], var_name: str
    ) -> torch.Tensor:
        s, b = self._get_scale(var_name)
        return (data - b) / s

    def _inv_scale(
        self, data: Union[np.ndarray, torch.Tensor], var_name: str
    ) -> torch.Tensor:
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
        data = []
        for var_name in self.config.input_variables:
            d = torch.from_numpy(lr[var_name]).to(self.dtype)
            d = self._scale(d, var_name)
            data.append(d)
            logger.debug(f"Input {var_name} shape = {d.shape}")
        data = torch.stack(data, dim=0)  # dim: channel, x, y

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

    def _make_segment_indices(self):
        x_indices = np.arange(
            0, self.config.x_full_size + 1, self.config.x_segmented_size
        )
        y_indices = np.arange(
            0, self.config.y_full_size + 1, self.config.y_segmented_size
        )

        dict_indices = {}
        n = 0
        for i, x_start in enumerate(x_indices[:-1]):
            x_end = x_indices[i + 1]
            for j, y_start in enumerate(y_indices[:-1]):
                y_end = y_indices[j + 1]
                dict_indices[n] = {
                    "x_slice": slice(x_start, x_end),
                    "y_slice": slice(y_start, y_end),
                }
                n += 1

        self.dict_indices = dict_indices

    def _extract_segmeted_data(self, data: torch.Tensor) -> torch.Tensor:
        if len(self.config.used_segment_indices) == 1:
            i = self.config.used_segment_indices[0]
        else:
            perm = torch.randperm(len(self.config.used_segment_indices))
            i = int(perm[0].item())
            i = self.config.used_segment_indices[i]
            logger.debug(f"segment index = {i}")

        x_slice = self.dict_indices[i]["x_slice"]
        y_slice = self.dict_indices[i]["y_slice"]
        logger.debug(f"{x_slice=}")
        logger.debug(f"{y_slice=}")

        return data[..., x_slice, y_slice]

    def __len__(self) -> int:
        return len(self.hr_files)

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
        assert target_data.shape[-2:] == input_data.shape[-2:] == s

        full_data = torch.cat([target_data, pos, input_data], dim=0)
        segmented = self._extract_segmeted_data(full_data)

        stacked = self.random_crop(segmented)

        n_pos = 4 if self.config.concat_topography_with_position else 2

        crop_target = stacked[0 : target_data.shape[0]]
        crop_pos = stacked[target_data.shape[0] : target_data.shape[0] + n_pos]
        crop_input = stacked[target_data.shape[0] + n_pos :]

        s = (target_data.shape[0], self.config.x_crop_size, self.config.y_crop_size)
        assert crop_target.shape == s

        s = (input_data.shape[0], self.config.x_crop_size, self.config.y_crop_size)
        assert crop_input.shape == s

        s = (n_pos, self.config.x_crop_size, self.config.y_crop_size)
        assert crop_pos.shape == s

        return dict(x=crop_input, pos=crop_pos, y=crop_target)
