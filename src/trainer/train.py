from __future__ import annotations
from typing import Dict, Any, Literal, Optional
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader


from .loops import train_one_epoch, validate
from .utils import build_binary_metrics, select_precision, save_checkpoint


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    device: str,
    epochs: int,
    checkpoint_dir: str,
    logger: Optional[Any] = None,
    use_amp: bool = False,
    use_bf16: bool = False,
) -> Dict[str, Any]:
    """
    Orchestrate the full training: precision setup, scaler, metrics, per-epoch logs, logging, checkpoints.

    Args:
        ;;;

    Returns:
        Dict[str, Any]: Summary with best F1 and checkpoint paths.
    """

    precision = select_precision(use_amp=use_amp, prefer_bf16=use_bf16)
    print(f"AMP precision: {precision}")

    scaler = GradScaler(enabled=(use_amp and precision == "fp16"))

    base = build_binary_metrics().to(device)
    train_metrics = base.clone()
    val_metrics = base.clone()
    best_val_f1: float = 0

    checkpoint_dir = Path(checkpoint_dir)
    best_path = checkpoint_dir / "best_model.pth"
    last_path = checkpoint_dir / "last_model.pth"

    for epoch in range(1, epochs):
        train_out = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            metrics=train_metrics,
            epoch=epoch,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            precision=precision,
            logger=logger,
        )

        val_out = validate(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            metrics=val_metrics,
            epoch=epoch,
            device=device,
            logger=logger,
        )

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train: loss={train_out['train/loss']:.4f}, acc={train_out['train/acc']:.4f}, f1={train_out['train/f1']:.4f} | "
            f"val:   loss={val_out['val/loss']:.4f}, acc={val_out['val/acc']:.4f}, f1={val_out['val/f1']:.4f}"
        )

        save_checkpoint(
            last_path,
            model.state_dict(),
            optimizer.state_dict(),
            epoch=epoch,
            precision=precision,
        )

        current_f1 = float(val_out["val/f1"])
        if current_f1 > best_val_f1:
            best_val_f1 = current_f1
            save_checkpoint(
                best_path,
                model.state_dict(),
                optimizer.state_dict(),
                epoch=epoch,
                precision=precision,
            )
            print(f"New best F1={best_val_f1:.4f} - saved {best_val_f1}")

    return {
        "best_val_f1": best_val_f1,
        "best_model_path": best_path,
        "last_model_path": last_path,
        "precision": precision,
    }
