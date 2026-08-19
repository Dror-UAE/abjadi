"""
Shared CUDA / CPU device helpers.

Keeps train / evaluate / predict consistent and CPU-compatible.
"""

from __future__ import annotations

from typing import Optional

import torch


def resolve_device(force_cpu: bool = False) -> torch.device:
    """Pick CUDA when available, otherwise CPU. ``force_cpu`` overrides detection."""
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_device(device: Optional[torch.device] = None) -> dict:
    """Return a serializable summary of the active compute device."""
    if device is None:
        device = resolve_device()
    info = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if device.type == "cuda" and torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(device)
    return info


def print_device_info(device: Optional[torch.device] = None, prefix: str = "") -> torch.device:
    """
    Print device details at startup.

    Example (CUDA):
      Device: cuda
      GPU: NVIDIA GeForce RTX 4060 Laptop GPU
      CUDA available: True
    """
    if device is None:
        device = resolve_device()
    info = describe_device(device)
    print(f"{prefix}Device: {info['device']}", flush=True)
    if info["cuda_available"] and device.type == "cuda":
        print(f"{prefix}GPU: {info['gpu_name']}", flush=True)
        print(f"{prefix}CUDA available: True", flush=True)
        if info["cuda_version"]:
            print(f"{prefix}CUDA (PyTorch build): {info['cuda_version']}", flush=True)
    elif info["cuda_available"] and device.type == "cpu":
        print(f"{prefix}CUDA available: True (forced CPU)", flush=True)
        print(f"{prefix}GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print(f"{prefix}CUDA available: False", flush=True)
    return device


def dataloader_kwargs(
    device: torch.device,
    *,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
) -> dict:
    """
    DataLoader options tuned for the active device.

    - ``pin_memory=True`` only when using CUDA (faster host→GPU copies)
    - ``persistent_workers`` when num_workers > 0
    """
    use_cuda = device.type == "cuda"
    workers = 0 if num_workers is None else int(num_workers)
    use_pin = use_cuda if pin_memory is None else bool(pin_memory)

    kwargs: dict = {
        "num_workers": workers,
        "pin_memory": use_pin and use_cuda,
    }
    if workers > 0:
        kwargs["persistent_workers"] = True
    return kwargs


def batch_to_device(
    images: torch.Tensor,
    labels: Optional[torch.Tensor],
    device: torch.device,
) -> tuple:
    """Move a training/eval batch to device (non_blocking when pinned CUDA)."""
    non_blocking = device.type == "cuda"
    images = images.to(device, non_blocking=non_blocking)
    if labels is None:
        return images, None
    labels = labels.to(device, non_blocking=non_blocking)
    return images, labels
