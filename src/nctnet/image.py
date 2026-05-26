from __future__ import annotations

import torch
from torch.utils.data import Dataset


def image_smoke_tensor(batch: int = 2, channels: int = 3, size: int = 16) -> torch.Tensor:
    return torch.zeros(batch, channels, size, size)


class SyntheticShapesDataset(Dataset):
    def __init__(self, size: int = 128, image_size: int = 16, channels: int = 3):
        self.size = size
        self.image_size = image_size
        self.channels = channels

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        h = w = self.image_size
        img = torch.zeros(self.channels, h, w)
        color = torch.zeros(self.channels)
        color[idx % self.channels] = 1.0
        x0 = 2 + (idx * 3) % max(1, w - 6)
        y0 = 2 + (idx * 5) % max(1, h - 6)
        if idx % 3 == 0:
            img[:, y0:y0 + 4, x0:x0 + 4] = color[:, None, None]
        elif idx % 3 == 1:
            img[:, y0:y0 + 2, :] = color[:, None, None]
            img[:, :, x0:x0 + 2] = color[:, None, None]
        else:
            for k in range(min(h, w)):
                img[:, k, (k + idx) % w] = color
        return {"image": img}
