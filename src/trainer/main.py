from __future__ import annotations
import torch
import torch.nn as nn
from torch.optim import AdamW

from src.data import build_loaders
from src.models import build_model
from src.trainer import train
import wandb


def main():
    """
    Main function to orchestrate the training process using Hydra configuration.
    """
    # print the configuration for reproducibility
    data_path = "/home/youness/python/AI-IN-MEDICAL-MATERIALS/Data/processed-pneumonia"
    batch_size = 4
    num_workers = 4
    lr = 1e-4
    max_epochs = 20
    use_amp = True
    use_bf16 = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # setup from config
    model = build_model(
        model_name="resnet18",
        num_classes=1,
        in_channels=1,
        weights=None,
    ).to(device)

    train_loader, val_loader, *_ = build_loaders(
        batch_size=batch_size, num_workers=num_workers, path=data_path
    )
    optimizer = AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    summary = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        epochs=max_epochs,
        logger=None,
        use_amp=use_amp,
        use_bf16=use_bf16,
    )
    print(summary)


if __name__ == "__main__":
    main()
