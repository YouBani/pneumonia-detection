# Pneumonia Detection with SageMaker (or Hydra)

This repository contains the training and hyperparameter tuning setup for pneumonia detection models.  
It is designed for **AWS SageMaker**, but also supports running locally using **Hydra** for configuration management.  
Experiment tracking is done with **Weights & Biases (W&B)**.  

## Repository Structure
```
├── configs/ # Hydra configs (training, model, data, hpo)
├── scripts/
│ ├── settings.py # Loads SageMaker config from environment
│ ├── sm_common.py # Shared SageMaker utilities (estimator, input, W&B key, metrics)
│ ├── sm_hpo.py # Entry point for SageMaker HPO
│ ├── sm_train.py # Entry point for SageMaker training
├── src/
├── src/
│   ├── trainer/
│   └── ...
├── tools/
│   ├── sm_entry.py           # The entry point for SageMaker containers.
│   └── ...
├── train.py # Hydra-based local training script
├── Dockerfile
├── docker-compose.yml # For running the training job locally with Docker.
├── requirements.docker.txt # Python dependencies for the Docker image.
└── README.md

pneumonia-detection/
├── configs/                   # Hydra configuration system
│   ├── config.yaml            # Default config
│   ├── train.yaml             # Training hyperparameters
│   ├── model.yaml             # Model configs
│   ├── data.yaml              # Dataset paths & transforms
│
├── scripts/                   # SageMaker entrypoints & common utils
│   ├── settings.py            # SMSettings loader (env-based)
│   ├── sm_common.py           # Shared SageMaker utilities (estimator, input, W&B key, metrics)
│   ├── sm_train.py            # SageMaker training job launcher
│   └── sm_hpo.py             # SageMaker hyperparameter tuning job launcher
│
├── src/                       # Core source code
│   ├── __init__.py
│   │
│   ├── data/                  # Datasets and dataloaders
│   │   ├── __init__.py
│   │   └── data.py            # Pneumonia dataset, dataloader, transforms
│   │
│   ├── models/                # Model architectures
│   │   ├── __init__.py
│   │   └── model.py           # Model architecture builder
│   │
│   └── trainer/               # Training logic
│       ├── __init__.py
│       ├── loops.py           # Training & validation loops
│       ├── main.py            # Hydra entrypoint for training
│       ├── train.py           # High-level training script
│       └── utils.py           # Helpers (logger, checkpoints, build_binary_metrics, amp_dtype_from_str, etc..)
│ 
├── tools/                     # Entrypoints for SageMaker containers
│   └── sm_entry.py            # Maps SageMaker env → Hydra config overrides
│
├── Dockerfile                 # Training image for SageMaker
├── docker-compose.yml         # For running the training job locally with Docker.
├── requirements.txt           # Python dependencies
├── requirements.docker.txt    # Python dependencies for the Docker image.
├── README.md
└── LICENSE
```
## Features

- **Hydra Mode**
  - Run locally with hierarchical configs.
  - Override parameters from CLI (`python train.py train.lr=1e-4 model.name=resnet34`).
- **SageMaker Mode**
  - Train and tune with AWS-managed infrastructure.
  - Bayesian HPO with early stopping.
- **Experiment Tracking** with W&B.
- **Reproducibility** via Docker and `.env`.

## Getting Started

### Clone the repo
```
git clone https://github.com/youbani/pneumonia-detection.git
cd pneumonia-detection
```
