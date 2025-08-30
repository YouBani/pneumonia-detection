from __future__ import annotations
import argparse
import os
import json
import shlex
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pathlib import Path
from typing import List, Any

from src.trainer.main import main as hydra_main


def _resolve_conf_dir() -> str:
    """Find absolute path to the Hydra configs folder."""
    base = Path(__file__).resolve().parent.parent
    return str((base / "configs").resolve())


def _parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments from the SageMaker training job.
    """
    parser = argparse.ArgumentParser("")
    parser.add_argument("--data-path", type=str, default=os.getenv("SM_CHANNEL_DATA"))
    parser.add_argument("--model-dir", type=str, default=os.getenv("SM_MODEL_DIR"))
    parser.add_argument("--extra-overrides", default="")

    args, unknown = parser.parse_known_args()
    return args


def _coarce_value(value: str) -> Any:
    """Coarce string values to their inferred type."""
    s = str(value).strip()
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        return int(s)
    except (ValueError, TypeError):
        pass
    try:
        return float(value)
    except (ValueError, TypeError):
        return s


ALLOWED_HPARAMS = {
    "data.path",
    "data.batch_size",
    "data.num_workers",
    "data.path",
    "train.max_epochs",
    "train.lr",
    "train.use_amp",
    "train.use_bf16",
    "train.checkpoint_dir",
    "model.name",
    "model.pretrained",
    "model.in_channels",
    "model.num_classes",
    "model.weights",
    "seed",
    "device",
}


def _overrides_from_hparams() -> List[str]:
    path = "/opt/ml/input/config/hyperparameters.json"
    try:
        with open(path) as f:
            hp = json.load(f)
    except FileNotFoundError:
        return []

    overrides = []
    for k, v in hp.items():
        if v in (None, ""):
            continue
        if k.startswith("_") or k.startswith("sagemaker_"):
            continue
        if ALLOWED_HPARAMS and k not in ALLOWED_HPARAMS:
            continue
        overrides.append(f"{k}={_coarce_value(v)}")
    return overrides


def _build_overrides(args) -> List[str]:
    """Map CLI + SageMaker env to the Hydra."""
    overrides = []

    overrides.extend(_overrides_from_hparams())

    data_path = args.data_path or "/opt/ml/input/data"
    model_dir = args.model_dir or "/opt/ml/model"

    overrides.append(f"data.path={data_path}")
    overrides.append(f"train.checkpoint_dir={model_dir}")

    if args.extra_overrides:
        overrides.extend(shlex.split(args.extra_overrides))

    return overrides


def main():
    args = _parse_args()
    overrides = _build_overrides(args)

    # Manually load the Hydra config and apply the overrides
    with initialize_config_dir(version_base="1.3", config_dir=_resolve_conf_dir()):
        cfg = compose(config_name="config", overrides=overrides)

    print("Effective config:\n" + OmegaConf.to_yaml(cfg), flush=True)

    if args.model_dir:
        Path(args.model_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.model_dir) / "config_used.yaml").write_text(OmegaConf.to_yaml(cfg))

    hydra_main(cfg)


if __name__ == "__main__":
    main()
