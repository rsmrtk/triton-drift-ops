"""
Synthetic drift transforms for GTSRB images — simulate the kind of input
distribution shift a real deployment would see (bad weather, low light,
sensor noise) without needing a second real-world dataset.

These plug into `training.data.dataset.get_loaders(extra_transform=...)`,
applied before resize/normalize, so the same loading code serves both the
clean baseline and every drift scenario.
"""

import random

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter


class Fog:
    """Blends the image toward a flat gray, simulating fog/haze."""

    def __init__(self, intensity: float = 0.5):
        self.intensity = intensity

    def __call__(self, img: Image.Image) -> Image.Image:
        gray = Image.new("RGB", img.size, (200, 200, 200))
        return Image.blend(img, gray, self.intensity)


class LowLight:
    """Darkens the image and reduces contrast, simulating night driving."""

    def __init__(self, brightness_factor: float = 0.35, contrast_factor: float = 0.7):
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor

    def __call__(self, img: Image.Image) -> Image.Image:
        img = TF.adjust_brightness(img, self.brightness_factor)
        img = TF.adjust_contrast(img, self.contrast_factor)
        return img


class SensorNoise:
    """Adds Gaussian noise, simulating a cheap/degraded camera sensor."""

    def __init__(self, std: float = 0.08):
        self.std = std

    def __call__(self, img: Image.Image) -> Image.Image:
        tensor = TF.to_tensor(img)
        noise = torch.randn_like(tensor) * self.std
        noisy = torch.clamp(tensor + noise, 0.0, 1.0)
        return TF.to_pil_image(noisy)


class MotionBlur:
    """Simulates a moving vehicle / out-of-focus capture."""

    def __init__(self, radius: float = 2.0):
        self.radius = radius

    def __call__(self, img: Image.Image) -> Image.Image:
        return img.filter(ImageFilter.GaussianBlur(radius=self.radius))


# Named scenarios used by drift experiments and the drift monitor's
# baseline-vs-scenario comparisons. Each maps to a torchvision-style
# transform (or composition) applied before resize/normalize.
DRIFT_SCENARIOS: dict[str, T.Compose] = {
    "clean": T.Compose([]),
    "fog": T.Compose([Fog(intensity=0.5)]),
    "night": T.Compose([LowLight(brightness_factor=0.35, contrast_factor=0.7)]),
    "noise": T.Compose([SensorNoise(std=0.08)]),
    "motion_blur": T.Compose([MotionBlur(radius=2.0)]),
    "severe": T.Compose(
        [Fog(intensity=0.35), LowLight(brightness_factor=0.55, contrast_factor=0.85), SensorNoise(std=0.04)]
    ),
}


def get_drift_transform(name: str) -> T.Compose:
    if name not in DRIFT_SCENARIOS:
        raise ValueError(f"unknown drift scenario '{name}', choices: {list(DRIFT_SCENARIOS)}")
    return DRIFT_SCENARIOS[name]


def random_scenario(exclude_clean: bool = True) -> str:
    choices = [k for k in DRIFT_SCENARIOS if not (exclude_clean and k == "clean")]
    return random.choice(choices)
