from dotenv import load_dotenv
import sagemaker

from scripts.sm_common import build_estimator, build_inputs, get_wandb_key
from scripts.settings import SMSettings


def main():
    """
    Launches a single training job on AWS SageMaker.
    """
    load_dotenv()
    cfg = SMSettings.from_env()

    sm_session = sagemaker.Session()

    wandb_api_key = get_wandb_key(cfg.secret_id, cfg.region)

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
        # "train.use_bf16": "true",
    }
    est = build_estimator(
        image_uri=cfg.image_uri,
        role=cfg.role,
        instance_type=cfg.instance_type,
        bucket=cfg.bucket,
        sagemaker_session=sm_session,
        base_hparams=hparams,
        env_extra=wandb_api_key,
        output_prefix=cfg.output_prefix,
    )

    est.fit(build_inputs(cfg.bucket), wait=False)

    job_name = est.latest_training_job.job_name
    print("Started job:", job_name)

    sm_session.logs_for_job(job_name=job_name, wait=True)
    print("Done. Model artifact at:", est.model_data)


if __name__ == "__main__":
    main()
