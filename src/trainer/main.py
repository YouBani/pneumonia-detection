from __future__ import annotations
import torch
import torch.nn as nn
from torch.optim import AdamW
from omegaconf import DictConfig, OmegaConf
import hydra
from typing import Dict, Any
import wandb

from src.data import build_loaders
from src.models import build_model
from src.trainer import train
from src.trainer.utils import _resolve_device


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Main function to orchestrate the training process using Hydra configuration.

    Args:
        cfg (DictConfig): The Hydra configuration object containing all the parameters.
    """
    # print the configuration for reproducibility
    print("Config:\n", OmegaConf.to_yaml(cfg, resolve=True))

    wandb.init(
        project="pneumonia-detection",
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    logger = wandb.run

    device = _resolve_device(cfg.device)
    print(f"Using device: {device}")

    # Build the model using parameters from the config
    model = build_model(
        model_name=cfg.model.name,
        num_classes=cfg.model.num_classes,
        in_channels=cfg.model.in_channels,
        weights=cfg.model.weights,
    ).to(device)

    # Build the data loaders using parameters from the config
    train_loader, val_loader, *_ = build_loaders(
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        path=cfg.data.path,
    )

    # Define optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=float(cfg.train.lr))
    loss_fn = nn.BCEWithLogitsLoss()

    # --- Train the Model ---
    summary: Dict[str, Any] = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        epochs=int(cfg.train.max_epochs),
        checkpoint_dir=cfg.train.checkpoint_dir,
        logger=logger,
        use_amp=bool(cfg.train.use_amp),
        use_bf16=bool(cfg.train.use_bf16),
    )
    print("\nTraining Summary:")
    print(summary)
    wandb.finish()


if __name__ == "__main__":
    main()
