import boto3
import json
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from typing import Dict, List, Optional
from scripts.settings import SMSettings


def get_wandb_key(cfg: SMSettings) -> Dict[str, str]:
    """
    Retrieves the raw secret from AWS Secrets Manager using boto3,
    and parses the JSON secret string to get the WANDB_API_KEY.
    """
    if not cfg.secret_id:
        return {}
    try:
        sm = boto3.client("secretsmanager", region_name=cfg.region)
        resp = sm.get_secret_value(SecretId=cfg.secret_id)
        secret_string = resp["SecretString"]
        return json.loads(secret_string)
    except json.JSONDecodeError:
        return secret_string
    except Exception as e:
        print(f"[wandb] Warning: failed to retrieve secret '{cfg.secret_id}': {e}")
        return {}


def build_inputs(cfg: SMSettings, prefix: str = "data") -> Dict[str, TrainingInput]:
    """
    Creates SageMaker TrainingInput channels for training and validation data.
    """
    return {
        "train": TrainingInput(
            s3_data=f"s3://{cfg.bucket}/{prefix}/train/", distribution="FullyReplicated"
        ),
        "val": TrainingInput(
            s3_data=f"s3://{cfg.bucket}/{prefix}/val/", distribution="FullyReplicated"
        ),
    }


def default_metric_defs() -> List[Dict[str, str]]:
    """
    Returns the metric definition for SageMaker to parse from the trainig logs.
    """
    return [
        {"Name": "val_loss", "Regex": r"val_loss:\s*([0-9.+-eE]+)"},
        {"Name": "val_acc", "Regex": r"val_acc:\s*([0-9.+-eE]+)"},
        {"Name": "val_f1", "Regex": r"val_f1:\s*([0-9.+-eE]+)"},
        {"Name": "val_auc", "Regex": r"val_auc:\s*([0-9.+-eE]+)"},
    ]


def build_estimator(
    cfg: SMSettings,
    base_hparams: Optional[Dict[str, str]] = None,
    env_extra: Optional[Dict[str, str]] = None,
    use_spot: bool = True,
    max_run: int = 36000,
    max_wait: int = 72000,
    metric_definitions: Optional[List[Dict[str, str]]] = None,
) -> Estimator:
    """
    Builds and returns a SageMaker Estimator object .

    Args:
        cfg (SMSettings): An instance of the SMSettings dataclass containing environment
                          configurations.
        base_hparams (Optional[Dict[str, str]]): A dictionary of hyperparameters for the
                                                 training job.
        env_extra (Optional[Dict[str, str]]): A dictionary of additional environment
                                              variables to pass to the training container.
        max_run (int): The maximum number of seconds to run the training job.
        max_wait (int): The maximum number of seconds to wait for a spot instance to
                        become available.
        metric_definitions (Optional[List[Dict[str, str]]]): A list of dictionaries
                                                              defining the regexes to
                                                              capture metrics from the
                                                              training logs.

    Returns:
        Estimator: The configured SageMaker Estimator object.
    """
    # Use a default set of hyperparameters if none are provided
    if base_hparams is None:
        base_hparams = {
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

    env = {
        "WANDB_DIR": "/opt/ml/model/wandb",
        **(env_extra or {}),
    }

    metrics = metric_definitions or default_metric_defs()
    return Estimator(
        image_uri=cfg.image_uri,
        role=cfg.role,
        instance_type=cfg.instance_type,
        instance_count=1,
        output_path=f"s3://{cfg.bucket}/{cfg.output_prefix}",
        use_spot_instances=use_spot,
        max_run=max_run,
        max_wait=max_wait,
        container_entry_point=["python", "-m", "tools.sm_entry"],
        hyperparameters=base_hparams,
        environment=env,
        metric_definitions=metrics,
        sagemaker_session=sagemaker.Session(),
    )
