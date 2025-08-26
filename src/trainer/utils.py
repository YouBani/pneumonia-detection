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


# ------ Logging ------
def _log(logger: Optional[Any], payload: Dict[str, Any]):
    """Log metrics if a logger with '.log()' is provided (e.g. wandb)."""
    if logger and hasattr(logger, "log"):
        logger.log(payload)


# ------ Metrics ------
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


# ------ Precision / AMP ------
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


# ------ Checkpoint ------
def save_checkpoint(
    path: str,
    model_state: Dict[str, Any],
    optimizer_state: Optional[Dict[str, Any]] = None,
    epoch: Optional[int] = None,
    precision: Optional[str] = None,
) -> None:
    """Save a checkpoint."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {"model": model_state}
    if optimizer_state is not None:
        payload["optimizer"] = optimizer_state
    if epoch is not None:
        payload["epoch"] = epoch
    if precision is not None:
        payload["precision"] = precision
    tmp = path_obj.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path_obj)
