from dotenv import load_dotenv
import sagemaker

from scripts.sm_common import (
    build_estimator,
    build_inputs,
    get_wandb_key,
    default_metric_defs,
)
from scripts.settings import SMSettings


def main():
    """
    Launches a single training job on AWS SageMaker.
    """
    load_dotenv()
    cfg = SMSettings.from_env()

    wandb_api_key = get_wandb_key(cfg)

    hparams = {
        "model.name": "resnet18",
        "model.in_channels": "1",
        "model.num_classes": "1",
        "train.lr": "1e-4",
        "data.batch_size": "32",
        "data.num_workers": "8",
        "train.max_epochs": "1",
        "device": "auto",
        "train.use_amp": "true",
    }
    est = build_estimator(
        cfg,
        base_hparams=hparams,
        env_extra=wandb_api_key,
        metric_definitions=default_metric_defs(),
    )
    est.fit(build_inputs(cfg), wait=False)

    job_name = est.latest_training_job.job_name
    print("Started job:", job_name)

    sagemaker.Session().logs_for_job(job_name=job_name, wait=True)
    print("Done. Model artifact at:", est.model_data)


if __name__ == "__main__":
    main()
