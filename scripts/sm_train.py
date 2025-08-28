import os
from dotenv import load_dotenv
import boto3
import json
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput

load_dotenv()

REGION = os.environ.get("AWS_REGION")
ROLE = os.environ.get("SAGEMAKER_ROLE")
BUCKET = os.environ.get("S3_BUCKET")
IMAGE_URI = os.environ.get("ECR_IMAGE_NAME")
INSTANCE_TYPE = os.environ.get("INSTANCE_TYPE")  # "ml.g4dn.xlarge"
SECRET_ID = os.environ.get("SECRET_ID")


def _get_wandb_key(secret_id: str, region: str) -> dict:
    """
    Retrieves the raw secret from AWS Secrets Manager using boto3.
    """
    sm = boto3.client("secretsmanager", region_name=region)
    resp = sm.get_secret_value(SecretId="pneumo/wandb")
    s = resp["SecretString"]
    return s


def _extract_key(secret_string: str) -> str:
    """
    Parses a JSON secret string to get the WANDB_API_KEY.
    """
    try:
        return json.loads(secret_string).get("WANDB_API_KEY", secret_string)
    except json.JSONDecodeError:
        return secret_string


def main():
    missing = [
        name
        for name, val in [
            ("SAGEMAKER_ROLE", ROLE),
            ("S3_BUCKET", BUCKET),
            ("ECR_IMAGE_NAME", IMAGE_URI),
        ]
        if not val
    ]
    if missing:
        raise ValueError(f"Missing required env var(s): {', '.join(missing)}")

    wandb_api_key = None
    if SECRET_ID:
        try:
            wandb_raw_secret = _get_wandb_key(SECRET_ID, REGION)
            wandb_api_key = _extract_key(wandb_raw_secret)
        except Exception as e:
            print(f"[wandb] Warning: failed to retrieve secret '{SECRET_ID}': {e}")

    inputs = {
        "train": TrainingInput(
            s3_data=f"s3://{BUCKET}/data/train/", distribution="FullyReplicated"
        ),
        "val": TrainingInput(
            s3_data=f"s3://{BUCKET}/data/val/", distribution="FullyReplicated"
        ),
    }

    # Metric regex for SM to parse logs printer by the main trainer
    metric_defs = [
        {"Name": "val_loss", "Regex": r"val_loss:\s*([0-9.+-eE]+)"},
        {"Name": "val_acc", "Regex": r"val_acc:\s*([0-9.+-eE]+)"},
        {"Name": "val_f1", "Regex": r"val_f1:\s*([0-9.+-eE]+)"},
        {"Name": "val_auc", "Regex": r"val_auc:\s*([0-9.+-eE]+)"},
    ]

    sagemaker_session = sagemaker.Session()

    est = Estimator(
        image_uri=IMAGE_URI,
        role=ROLE,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        output_path=f"s3://{BUCKET}/pneumo/output/",
        use_spot_instances=True,
        max_run=36000,
        max_wait=72000,
        container_entry_point=["python", "-m", "tools.sm_entry"],
        hyperparameters={
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
        },
        environment={
            "WANDB_DIR": "/opt/ml/model/wandb",
            "WANDB_API_KEY": wandb_api_key,
        },
        metric_definitions=metric_defs,
        sagemaker_session=sagemaker_session,
    )

    # Submit and stream logs until it finishes
    est.fit(inputs, wait=False)
    job_name = est.latest_training_job.job_name
    print("Started job:", job_name)
    sagemaker.Session().logs_for_job(job_name=job_name, wait=True)
    print("Done. Model artifact at:", est.model_data)


if __name__ == "__main__":
    main()
