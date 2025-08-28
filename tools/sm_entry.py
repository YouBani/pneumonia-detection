from __future__ import annotations
import argparse
import os
import json
import shlex
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pathlib import Path
from typing import List

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


def _overrides_from_hparams() -> List[str]:
    path = "/opt/ml/input/config/hyperparameters.json"
    try:
        with open(path) as f:
            hp = json.load(f)
    except FileNotFoundError:
        return []
    return [f"{k}={v}" for k, v in hp.items() if v not in (None, "")]


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
