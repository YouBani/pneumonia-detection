FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.docker.txt /workspace/requirements.docker.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.docker.txt


COPY . /workspace

ENV WANDB_PROJECT=pneumonia-detection
ENV WANDB_DIR=/workspace/wandb

CMD ["python", "-m", "src.trainer.main"]