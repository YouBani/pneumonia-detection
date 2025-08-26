from __future__ import annotations

from typing import Dict, Literal, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryAUROC,
)
from tqdm import tqdm

from .utils import _log, amp_dtype_from_str


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    metrics: MetricCollection,
    epoch: int,
    device: str,
    scaler: Optional[GradScaler | None] = None,
    use_amp: bool = False,
    precision: Literal["bf16", "fp16", "fp32"] = "fp16",
    logger=None,
) -> Dict[str, float]:
    """
    Executes a single training epoch with AMP (bf16/fp16).

    Args:
        model (nn.Module): The PyTorch model to train.
        loader (DataLoader): DataLoader for the training data.
        optimizer (Optimizer): Optimizer for updating the model weights.
        loss_fn (nn.Module): The loss function to compute the training loss.
        metrics (MetricCollection): A MetricCollection instance to compute all metrics.
        epoch (int): The current epoch number.
        device (str): Device string (e.g. "cuda" or "cpu).
        scaler (GradScaler | None): The AMP scaler for gradient scaling.
        use_amp (bool | None): Whether to use Automatic Mixed Precision.
        precision (Literal): Precision type ("bf16", "fp16", "fp32").
        logger: The logger to use to log the metrics.

    Returns:
        Dict[str, float]: Dictionary with metrics, avg loss, and epoch.
    """
    if len(loader) == 0:
        raise ValueError("Empty training DataLoader passed to train_one_epoch.")

    model.train()
    metrics.reset()

    total_loss = 0.0
    total_items = 0

    amp_dtype = amp_dtype_from_str(precision)
    progress_bar = tqdm(loader, desc=f"Train Epoch {epoch}", leave=False)

    for step, (x, y) in enumerate(progress_bar, 1):
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", enabled=use_amp, dtype=amp_dtype):
            logits = model(x).squeeze(1)
            loss = loss_fn(logits, y)

        if scaler is not None and use_amp and amp_dtype == torch.float16:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            preds = torch.sigmoid(logits).detach()
            metrics.update(preds, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_items += bs

        progress_bar.set_postfix(loss=loss.item(), acc=metrics["acc"].compute().item())

        _log(logger, {"train/loss_step": loss.item(), "epoch": epoch, "step": step})

    avg_loss = total_loss / total_items
    computed = metrics.compute()
    metrics_dict = {f"train/{k}": v.item() for k, v in computed.items()}
    metrics_dict.update({"train/loss": avg_loss, "epoch": epoch})

    _log(logger, metrics_dict)
    return metrics_dict


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    metrics: MetricCollection,
    epoch: int,
    device: str,
    logger=None,
):
    """
    Evaluates the model on the validation data.

    Args:
        model (nn.Module): The PyTorch model to train.
        loader (DataLoader): DataLoader for the validation data.
        loss_fn (nn.Module): The loss function to compute the validation loss.
        metrics (MetricCollection): A MetricCollection instance to compute all metrics.
        epoch (int): The current epoch number.
        device (str): Device string ("cuda" or "cpu").


    Returns:
        Dict[str, float]: Dictionary with metrics, avg loss, and epoch.
    """
    if len(loader) == 0:
        raise ValueError("Empty DataLoader passed to validate().")

    model.eval()
    metrics.reset()

    total_loss = 0.0
    total_items = 0

    progress_bar = tqdm(loader, desc=f"Val Epoch {epoch}", leave=False)

    for x, y in progress_bar:
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).float()

        logits = model(x).squeeze(1)
        loss = loss_fn(logits, y)

        preds = torch.sigmoid(logits)
        metrics.update(preds, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_items += bs
        progress_bar.set_postfix(loss=loss.item(), acc=metrics["acc"].compute().item())

    avg_loss = total_loss / total_items
    computed = metrics.compute()
    metrics_dict = {f"val/{k}": v.item() for k, v in computed.items()}
    metrics_dict["val/loss"] = avg_loss
    metrics_dict["epoch"] = epoch

    _log(logger, metrics_dict)
    return metrics_dict
