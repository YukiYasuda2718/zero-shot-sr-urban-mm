#!/bin/bash

# Run this script to train a model inside a Docker container.

ROOT_DIR=/workspace
WORLD_SIZE=1
SCRIPT_PATH=$ROOT_DIR/python/scripts/train_ddp_model.py
CONFIG_PATH=$ROOT_DIR/python/configs/ZeroShotPointCloudData/tno_s25097_rT.yaml

python3 $SCRIPT_PATH --config_path $CONFIG_PATH --world_size $WORLD_SIZE
