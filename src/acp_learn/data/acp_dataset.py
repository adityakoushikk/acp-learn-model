"""PyTorch Dataset for ACP feature matrix + labels."""
import torch
from torch.utils.data import Dataset
import numpy as np


class ACPDataset(Dataset):
    """Dataset of (X, y) for ACP classification. X: float tensor, y: long 0/1."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
