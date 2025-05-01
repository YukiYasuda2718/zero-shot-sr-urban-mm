
# Description

This repository contains the source code used in "Zero-Shot Super-Resolution from Unstructured Data Using a Transformer-Based Neural Operator for Urban Micrometeorology" by Yuki Yasuda and Ryo Onishi. We have used PyTorch 1.12.1 with an NVIDIA RTX A6000.
- [Link to arXiv](https://arxiv.org/abs/2504.21361)

# Files and Directories

```txt
├── .devcontainer                    # for VSCode users
├── LICENSE
├── README.md
├── data
├── docker
│   └── pytorch_v1
├── docker-compose.yml               # for non-VSCode users
└── python
    ├── bash
    │   └── train_model.sh           # main script to train model, which runs train_ddp_model.py
    ├── configs
    │   ├── ZeroShotPointCloudData   # configs using unstructured data (point cloud data)
    │   └── ZeroShotUnseenData       # configs using structured data (equidistant gridded data)
    ├── scripts
    │   └── train_ddp_model.py
    └── src
        ├── configs
        ├── data                     # custom datasets and dataloaders
        ├── models                   # CNNs and transformer-based neural operators
        └── utils
```
