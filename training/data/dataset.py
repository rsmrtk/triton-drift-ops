"""
GTSRB dataset loading, with a clean seam for injecting drift transforms
(fog/night/noise) in the drift-simulation stage without touching the
training loop.
"""

from pathlib import Path

import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import GTSRB

from model.net import INPUT_SIZE

NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

BASE_TRANSFORM = T.Compose(
    [
        T.Resize((INPUT_SIZE, INPUT_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ]
)


def get_loaders(
    data_dir: str = "./gtsrb-data",
    batch_size: int = 64,
    num_workers: int = 2,
    extra_transform: T.Compose | None = None,
) -> tuple[DataLoader, DataLoader]:
    """
    extra_transform is applied before resize/normalize — this is the hook
    the drift stage uses to inject fog/night/noise degradation without
    duplicating the loading logic.
    """
    transform = BASE_TRANSFORM
    if extra_transform is not None:
        transform = T.Compose([extra_transform, BASE_TRANSFORM])

    Path(data_dir).mkdir(parents=True, exist_ok=True)

    train_set = GTSRB(root=data_dir, split="train", download=True, transform=transform)
    test_set = GTSRB(root=data_dir, split="test", download=True, transform=transform)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader
