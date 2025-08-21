import torch
import random, numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision import datasets


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loaders(cfg):
    """Build train/val dataloaders from Hydra cfg.dataset"""
    mean, std = cfg.dataset.mean, cfg.dataset.std

    train_tfm = T.Compose(
        [
            T.RandomResizedCrop((224, 224), scale=(0.35, 1)),
            T.RandomAffine(translate=(0, 0.05), scale=(0.9, 1.1), degrees=(-5, 5)),
            T.ToTensor(),
            T.Normalize(mean=[mean], std=[std]),
        ]
    )

    val_tfm = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[mean], std=[std]),
        ]
    )

    train_ds = datasets.ImageFolder(
        root=f"{cfg.dataset.path}/train", transform=train_tfm
    )
    val_ds = datasets.ImageFolder(root=f"{cfg.dataset.path}/val", transform=val_tfm)

    g = torch.Generator()
    g.manual_seed(cfg.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    return train_loader, val_loader, train_ds, val_ds
