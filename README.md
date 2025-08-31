# Pneumonia Detection with SageMaker (or Hydra)

This repository contains the training and hyperparameter tuning setup for pneumonia detection models.  
It is designed for **AWS SageMaker**, but also supports running locally using **Hydra** for configuration management.  
Experiment tracking is done with **Weights & Biases (W&B)**.  

## Repository Structure
```
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

- **Hydra Mode** (local)
  - Run locally with hierarchical configs.
  - Override parameters from CLI (`python -m src.trainer.main.py train.lr=1e-4 model.name=resnet34`).
 
- **SageMaker Mode** (cloud)
  - Train and tune with AWS-managed infrastructure.
  - Bayesian HPO with early stopping.

- **Experiment Tracking** with W&B.
- **Reproducibility** via Docker and `.env`.

# Part 1) Local: Conda + Hydra

### 1. Create and activate the Conda env
```
conda env create -f environment.yml
conda activate pneumo-py310
```
### 2. Set W&B (Optional)
```
export WANDB_API_KEY="<your_wandb_key>"
```
### 3. Run a local training job with Hydra
```
# Default config
python -m src.trainer.main

# Override hyperparameters from CLI
python -m src.trainer.main train.lr=1e-4 data.batch_size=64 model.name=resnet34
```
# Part 2) AWS SageMaker
### 1. Prereqs
* An S3 bucket for inputs and model artifacts (e.g., s3://my-sagemaker-bucket).
* A SageMaker execution role with permissions for S3 and Secrets Manager.
* Your Docker image built/pushed if you use a custom image.

### 2. Set up environment variables
Create a .env file in the root of the repository to configure your SageMaker and S3 settings.
```
# AWS Credentials and SageMaker Role
AWS_REGION=<aws-region>
SAGEMAKER_ROLE=<sagemaker-execution-role>

# Docker Image and S3 Bucket
S3_BUCKET=<sagemaker-bucket>
ECR_IMAGE_NAME=<image_path>

# Instance Type and W&B Secret ID
INSTANCE_TYPE=<ml.g4dn.xlarge>
SECRET_ID=<pneumo/wandb>
```

### 3. Store W&B API key in AWS Secrets Manager
Create the secret:
```
aws secretsmanager create-secret \
  --name pneumonia/wandb \
  --secret-string '{"WANDB_API_KEY":"<your_api_key>"}' \
  --region <yourregion>
```
### 4. Start a training job
```
python scripts/sm_train.py
```
### 5. Run Hyperparameter Tuning
```
python scripts/sm_hpo.py
```
Typical search space (example):
* LR: 1e-5 → 3e-3 (log)
* Batch size: [32, 64, 128]
* Model: [resnet18, resnet34, resnet50]
* Epochs: 30 → 40
Objective metric: maximize validation AUC.

# Notes
* Mixed precision is enabled by default via train.use_amp=true.
* Training writes checkpoints to train.checkpoint_dir (set by SageMaker to the model dir).
* Logs go to W&B and CloudWatch.
* tools/sm_entry.py is the on-container entrypoint that composes Hydra config with SageMaker-provided overrides.


