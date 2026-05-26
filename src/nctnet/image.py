from __future__ import annotations

import torch


def image_smoke_tensor(batch: int = 2, channels: int = 3, size: int = 16) -> torch.Tensor:
    return torch.zeros(batch, channels, size, size)
