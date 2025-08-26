import torch
import random
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision import datasets


__all__ = ["build_loaders"]


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _load_file(path):
    return np.load(path).astype(np.float32)


def build_loaders(batch_size, num_workers, path):
    """Build train/val dataloaders."""
    mean, std = 0.0853, 0.2340

    train_tfm = T.Compose(
        [
            T.ToTensor(),
            T.RandomResizedCrop((224, 224), scale=(0.35, 1)),
            T.RandomAffine(translate=(0, 0.05), scale=(0.9, 1.1), degrees=(-5, 5)),
            T.Normalize(mean=[mean], std=[std]),
        ]
    )

    val_tfm = T.Compose(
        [
            T.ToTensor(),
            T.Resize((224, 224)),
            T.Normalize(mean=[mean], std=[std]),
        ]
    )

    train_ds = datasets.DatasetFolder(
        root=f"{path}/train",
        loader=_load_file,
        extensions=("npy",),
        transform=train_tfm,
    )
    val_ds = datasets.DatasetFolder(
        root=f"{path}/val",
        loader=_load_file,
        extensions=("npy",),
        transform=val_tfm,
    )

    g = torch.Generator().manual_seed(42)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_seed_worker,
        generator=g,
    )

    return train_loader, val_loader, train_ds, val_ds
