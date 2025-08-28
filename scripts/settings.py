import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class SMSettings:
    region: str
    role: str
    bucket: str
    image_uri: str
    instance_type: str
    secret_id: Optional[str]
    output_prefix: str = "pneumo/output/"

    @classmethod
    def from_env(cls) -> "SMSettings":
        """Builds SMSettings from environment variables."""
        e = os.environ
        region = e.get("AWS_REGION", "")
        role = e.get("SAGEMAKER_ROLE", "")
        bucket = e.get("S3_BUCKET", "")
        image_uri = e.get("ECR_IMAGE_NAME", "")
        instance_type = e.get("INSTANCE_TYPE", "ml.m5.xlarge")
        wandb_secret = e.get("SECRET_ID", "")
        output_prefix = e.get("OUTPUT_PREFIX", "pneumo/output/")

        missing = [
            n
            for n, v in [
                ("AWS_REGION", region),
                ("SAGEMAKER_ROLE", role),
                ("S3_BUCKET", bucket),
                ("ECR_IMAGE_NAME", image_uri),
                ("INSTANCE_TYPE", instance_type),
            ]
            if not v
        ]
        if missing:
            raise ValueError(f"Missing required env var(s): {', '.join(missing)}")
        return cls(
            region=region,
            role=role,
            bucket=bucket,
            image_uri=image_uri,
            instance_type=instance_type,
            output_prefix=output_prefix,
            secret_id=wandb_secret,
        )
