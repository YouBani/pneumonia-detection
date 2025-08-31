import time
import sys
from dotenv import load_dotenv
import sagemaker
from sagemaker.tuner import (
    HyperparameterTuner,
    ContinuousParameter,
    IntegerParameter,
    CategoricalParameter,
)

from scripts.settings import SMSettings
from scripts.sm_common import (
    get_wandb_key,
    build_estimator,
    build_inputs,
    default_metric_defs,
)


def main():
    load_dotenv()

    cfg = SMSettings.from_env()

    wandb_api_key = get_wandb_key(cfg)

    base_hparams = {
        "model.name": "resnet18",
        "model.in_channels": "1",
        "model.num_classes": "1",
        "data.num_workers": "8",
        "device": "auto",
        "train.use_amp": "true",
        "train.max_epochs": "10",
    }

    est = build_estimator(
        cfg,
        base_hparams=base_hparams,
        env_extra=wandb_api_key,
        metric_definitions=default_metric_defs(),
    )
    ranges = {
        "train.lr": ContinuousParameter(1e-5, 3e-3, scaling_type="Logarithmic"),
        "data.batch_size": CategoricalParameter(["32", "64", "128"]),
        "model.name": CategoricalParameter(["resnet18", "resnet34", "resnet50"]),
        "train.max_epochs": IntegerParameter(30, 40),
    }

    tuner = HyperparameterTuner(
        estimator=est,
        objective_metric_name="val_auc",
        objective_type="Maximize",
        hyperparameter_ranges=ranges,
        metric_definitions=default_metric_defs(),
        max_jobs=12,
        max_parallel_jobs=3,
        early_stopping_type="Auto",
        strategy="Bayesian",
    )

    job_name = f"pneumo-hpo-{int(time.time())}"
    tuner.fit(
        inputs=build_inputs(cfg),
        include_cls_metadata=False,
        job_name=job_name,
        wait=False,
    )

    print("Started tuning job:", job_name)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[HPO] ERROR: {e}", file=sys.stderr, flush=True)
        raise
