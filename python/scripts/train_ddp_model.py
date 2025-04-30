import argparse
import datetime
import os
import pathlib
import sys
import time
import traceback
from logging import INFO, FileHandler, StreamHandler, getLogger

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from src.configs.base_config import BaseExperimentConfig
from src.configs.config_loader import load_config
from src.data.dataloader import make_dataloaders_and_samplers
from src.models.loss_maker import make_loss
from src.models.model_maker import make_model
from src.models.optim_helper import optimize_ddp
from src.utils.random_seed_helper import set_seeds
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ["CUBLAS_WORKSPACE_CONFIG"] = r":4096:8"  # to make calculations deterministic
set_seeds(42, use_deterministic=True)

logger = getLogger()
logger.addHandler(StreamHandler(sys.stdout))
logger.setLevel(INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, required=True)
parser.add_argument("--world_size", type=int, required=True)

ROOT_DIR = str((pathlib.Path(os.environ["PYTHONPATH"]) / "..").resolve())


def setup(rank: int, world_size: int, backend: str = "nccl"):
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train_and_validate(
    rank: int,
    world_size: int,
    config: BaseExperimentConfig,
    result_dir_path: str,
):
    setup(rank, world_size)

    dataloaders, samplers = make_dataloaders_and_samplers(
        root_dir=ROOT_DIR,
        config_data=config.data,
        config_loader=config.loader,
        world_size=world_size,
        rank=rank,
    )

    set_seeds(config.train.seed)
    logger.info(f"Seed has been set: seed = {config.train.seed}")
    model = make_model(config.model)
    model = DDP(model.to(rank), device_ids=[rank])

    loss_fn = make_loss(config.train.loss_name)

    optimizer = ZeroRedundancyOptimizer(
        model.parameters(),
        optimizer_class=torch.optim.AdamW,
        lr=config.train.learning_rate,
    )

    all_scores = []
    best_epoch = 0
    best_loss = np.inf
    es_cnt = 0  # for early stopping

    weight_path = f"{result_dir_path}/model_weight.pth"
    learning_history_path = f"{result_dir_path}/model_loss_history.csv"

    if rank == 0:
        logger.info("\n###############################")
        logger.info(f"Train model: Seed = {config.train.seed}")
        logger.info("###############################\n")

    for epoch in range(config.train.epochs):
        _time = time.time()
        losses = {}

        if rank == 0:
            logger.info(f"Epoch: {epoch + 1} / {config.train.epochs}")

        for mode in ["train", "valid"]:
            dist.barrier()
            loss = optimize_ddp(
                dataloader=dataloaders[mode],
                sampler=samplers[mode],
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epoch=epoch,
                rank=rank,
                world_size=world_size,
                mode=mode,
            )
            losses[mode] = loss
            dist.barrier()

        all_scores.append(losses)

        if losses["valid"] > best_loss:
            es_cnt += 1
            if rank == 0:
                logger.info(f"ES count = {es_cnt}")
            if es_cnt >= config.train.early_stopping_patience:
                break
        else:
            best_epoch = epoch + 1
            best_loss = losses["valid"]
            es_cnt = 0

            if rank == 0:
                torch.save(model.module.state_dict(), weight_path)
                logger.info(
                    "Best loss is updated, ES count is reset, and model weights are saved."
                )

        if rank == 0:
            if epoch % 10 == 0:
                pd.DataFrame(all_scores).to_csv(learning_history_path, index=False)
            logger.info(f"Train loss = {losses['train']:.8f}")
            logger.info(f"Valid loss = {losses['valid']:.8f}")
            logger.info(f"Elapsed time = {time.time() - _time} sec")
            logger.info("-----")
            # logger.info(torch.cuda.memory_summary(device=rank))

    if rank == 0:
        pd.DataFrame(all_scores).to_csv(learning_history_path, index=False)
        logger.info(f"Best epoch: {best_epoch}, best_loss: {best_loss:.8f}")

    cleanup()


if __name__ == "__main__":
    try:
        os.environ["MASTER_ADDR"] = "localhost"

        # Port is arbitrary, but set random value to avoid collision
        np.random.seed(datetime.datetime.now().microsecond)
        port = str(np.random.randint(12000, 65535))
        os.environ["MASTER_PORT"] = port

        world_size = parser.parse_args().world_size
        config_path = parser.parse_args().config_path

        experiment_name = config_path.split("/")[-2]
        config_name = os.path.basename(config_path).split(".")[0]

        config = load_config(experiment_name=experiment_name, config_path=config_path)

        result_dir_path = f"{ROOT_DIR}/data/models/{experiment_name}/{config_name}"
        os.makedirs(result_dir_path, exist_ok=False)

        logger.addHandler(FileHandler(f"{result_dir_path}/log.txt"))

        logger.info("\n*********************************************************")
        logger.info(f"Start DDP: {datetime.datetime.utcnow()} UTC.")
        logger.info("*********************************************************\n")

        logger.info(f"experiment name = {experiment_name}")
        logger.info(f"config name = {config_name}")
        logger.info(f"config path = {config_path}")
        logger.info(f"\nInput config = {config.to_json_str()}\n")

        if not torch.cuda.is_available():
            logger.error("No GPU.")
            raise Exception("No GPU.")

        logger.info(f"Num available GPUs = {torch.cuda.device_count()}")
        logger.info(f"Names of GPUs = {torch.cuda.get_device_name()}")
        logger.info(f"Device capability = {torch.cuda.get_device_capability()}")
        logger.info(f"World size = {world_size}")
        logger.info(torch.cuda.memory_summary())

        start_time = time.time()

        mp.spawn(
            train_and_validate,
            args=(world_size, config, result_dir_path),
            nprocs=world_size,
            join=True,
        )

        end_time = time.time()

        logger.info(
            f"Total elapsed time for training = {(end_time - start_time) / 60.} min"
        )

        logger.info("\n*********************************************************")
        logger.info(f"End DDP: {datetime.datetime.utcnow()} UTC.")
        logger.info("*********************************************************\n")

    except Exception as e:
        logger.info("\n*********************************************************")
        logger.info("Error")
        logger.info("*********************************************************\n")
        logger.error(e)
        logger.error(traceback.format_exc())
