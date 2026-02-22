"""MLP architectures for ACP classification."""
import torch
import torch.nn as nn


class DeepMLP(nn.Module):
    """Multi-layer MLP with dropout. Matches notebook architecture."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        dropout_rates: list[float],
        activation: str = "relu",
    ):
        super().__init__()
        act = getattr(nn, activation.capitalize(), nn.ReLU) if hasattr(nn, activation.capitalize()) else nn.ReLU
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(act())
            if i < len(dropout_rates):
                layers.append(nn.Dropout(dropout_rates[i]))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return self.head(x).squeeze(-1)
