import copy
import dataclasses
import glob
import os
from logging import getLogger

from sklearn.model_selection import train_test_split
from src.data.datasets.base_config import BaseConfigDataset
from src.data.datasets.dataset_segmented_tuv_with_pos import DatasetSegmentedTUVWithPos
from src.data.datasets.dataset_tuv_point_cloud import DatasetPointCloudTUVWithPos
from src.utils.base_config import YamlConfig
from src.utils.random_seed_helper import get_torch_generator, seed_worker
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

logger = getLogger()


def split_paths_into_train_valid_test(
    paths: list[str], train_valid_test_ratios: list[float]
) -> tuple[list[str], list[str], list[str]]:
    #
    logger.info(f"train, valid, test ratios = {train_valid_test_ratios}")

    assert len(train_valid_test_ratios) == 3  # train, valid, test, three ratios
    assert sum(train_valid_test_ratios) == 1.0
    assert all([r > 0 for r in train_valid_test_ratios])

    test_size = train_valid_test_ratios[-1]
    _paths, test_paths = train_test_split(paths, test_size=test_size, shuffle=False)

    valid_size = train_valid_test_ratios[1] / (
        train_valid_test_ratios[0] + train_valid_test_ratios[1]
    )
    train_paths, valid_paths = train_test_split(
        _paths, test_size=valid_size, shuffle=False
    )

    assert set(train_paths).isdisjoint(set(valid_paths))
    assert set(train_paths).isdisjoint(set(test_paths))
    assert set(valid_paths).isdisjoint(set(test_paths))

    logger.info(
        f"train: {len(train_paths)}, valid: {len(valid_paths)}, test: {len(test_paths)}"
    )

    return train_paths, valid_paths, test_paths


def _make_dataloaders_and_samplers(
    *,
    dataset_initilizer,
    dict_configs: dict[str, YamlConfig],
    train_valid_test_kinds: list[str],
    batch_size: int,
    world_size: int = None,
    rank: int = None,
    num_workers: int = 2,
    seed: int = 42,
    **kwargs,
):
    logger.info(
        f"batch size = {batch_size}, world_size = {world_size}, rank = {rank}, num_workers = {num_workers}, seed = {seed}\n"
    )

    if world_size is not None:
        assert isinstance(rank, int)
        assert batch_size % world_size == 0, "batch_size % world_size /= 0."

    dict_dataloaders, dict_samplers = {}, {}

    for kind in train_valid_test_kinds:
        logger.info(f"\n{kind} dataloader and sampler is being made:")
        dataset = dataset_initilizer(dict_configs[kind])

        if world_size is None:
            dict_dataloaders[kind] = DataLoader(
                dataset,
                batch_size=batch_size,
                drop_last=True if kind == "train" else False,
                shuffle=True if kind == "train" else False,
                pin_memory=True,
                num_workers=num_workers,
                worker_init_fn=seed_worker,
                generator=get_torch_generator(),
            )
            logger.info(
                f"{kind}: dataset size = {len(dict_dataloaders[kind].dataset)}, batch num = {len(dict_dataloaders[kind])}\n"
            )

        else:
            dict_samplers[kind] = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                seed=seed,
                shuffle=True if kind == "train" else False,
                drop_last=True if kind == "train" else False,
            )

            dict_dataloaders[kind] = DataLoader(
                dataset,
                sampler=dict_samplers[kind],
                batch_size=batch_size // world_size,
                pin_memory=True,
                num_workers=num_workers,
                worker_init_fn=seed_worker,
                generator=get_torch_generator(),
                drop_last=True if kind == "train" else False,
            )

            if rank == 0:
                logger.info(
                    f"{kind}: dataset size = {len(dict_dataloaders[kind].dataset)}, batch num = {len(dict_dataloaders[kind])}\n"
                )
    return dict_dataloaders, dict_samplers


@dataclasses.dataclass()
class ConfigDataloader(YamlConfig):
    batch_size: int
    dl_data_name: str
    train_valid_test_ratios: list[float]
    dataset_name: str


def make_dataloaders_and_samplers(
    root_dir: str,
    config_loader: ConfigDataloader,
    config_data: BaseConfigDataset,
    train_valid_test_kinds: list[str] = ["train", "valid", "test"],
    world_size: int = None,
    rank: int = None,
):
    logger.info(f"\nInput dataloader config = {config_loader.to_json_str()}")

    dl_data_dir = f"{root_dir}/data/processed/DL_data/{config_loader.dl_data_name}"

    data_dirs = sorted(
        [path for path in glob.glob(f"{dl_data_dir}/*") if os.path.isdir(path)]
    )

    train_dirs, valid_dirs, test_dirs = split_paths_into_train_valid_test(
        data_dirs, config_loader.train_valid_test_ratios
    )
    _train = copy.deepcopy(config_data)
    _train.dir_paths = train_dirs

    _valid = copy.deepcopy(config_data)
    _valid.dir_paths = valid_dirs

    _test = copy.deepcopy(config_data)
    _test.dir_paths = test_dirs

    dict_config_datas = {"train": _train, "valid": _valid, "test": _test}

    if config_loader.dataset_name == "DatasetSegmentedTUVWithPos":
        dataset_initilizer = DatasetSegmentedTUVWithPos
        logger.info("Dataset is DatasetSegmentedTUVWithPos")
    elif config_loader.dataset_name == "DatasetPointCloudTUVWithPos":
        dataset_initilizer = DatasetPointCloudTUVWithPos
        logger.info("Dataset is DatasetPointCloudTUVWithPos")
    else:
        raise NotImplementedError(f"{config_loader.dataset_name} is not supported.")

    return _make_dataloaders_and_samplers(
        dataset_initilizer=dataset_initilizer,
        dict_configs=dict_config_datas,
        batch_size=config_loader.batch_size,
        train_valid_test_kinds=train_valid_test_kinds,
        world_size=world_size,
        rank=rank,
    )
