import os
import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryAUROC,
)
from pathlib import Path
from typing import Optional, Dict, Any, Literal


def _resolve_device(requested: str = "auto") -> str:
    """
    Resolve device string based on requested string.

    Args:
        requested (str): "auto", "cpu", "cuda", "mps".

    Returns:
        str: The resolved device.
    """
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "mps" and torch.backends.mps.is_available():
        return "mps"
    return requested


def _log(logger: Optional[Any], payload: Dict[str, Any]):
    """Log metrics if a logger with '.log()' is provided (e.g. wandb)."""
    if logger and hasattr(logger, "log"):
        logger.log(payload)


def build_binary_metrics() -> MetricCollection:
    """Standard binary classifiction metrics."""
    return MetricCollection(
        {
            "acc": BinaryAccuracy(),
            "precision": BinaryPrecision(),
            "recall": BinaryRecall(),
            "f1": BinaryF1Score(),
            "auroc": BinaryAUROC(),
        }
    )


def select_precision(
    use_amp: bool, prefer_bf16: bool
) -> Literal["bf16", "fp16", "fp32"]:
    """Decide global AMP precision"""
    if not use_amp:
        return "fp16"
    elif use_amp and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return "bf16"
    return "fp32"


def amp_dtype_from_str(precision: Literal["bf16", "fp16", "fp32"]):
    """Maps the precision string to torch dtype used by autocast."""
    if precision == "bf16":
        return torch.bfloat16
    elif precision == "fp16":
        return torch.float16
    return torch.float32


def save_checkpoint(
    path: str,
    model_state: Dict[str, Any],
    optimizer_state: Optional[Dict[str, Any]] = None,
    epoch: Optional[int] = None,
    precision: Optional[str] = None,
) -> None:
    """Save a checkpoint to a file."""
    dest = Path(path)
    assert not dest.is_dir(), f"Checkpoint path points to a directory {dest}"

    payload: Dict[str, Any] = {"model": model_state}
    if optimizer_state is not None:
        payload["optimizer"] = optimizer_state
    if epoch is not None:
        payload["epoch"] = epoch
    if precision is not None:
        payload["precision"] = precision

    tmp = (
        dest.with_suffix(dest.suffix + ".tmp")
        if dest.suffix
        else dest.with_suffix(".tmp")
    )
    torch.save(payload, tmp)
    os.replace(tmp, dest)
