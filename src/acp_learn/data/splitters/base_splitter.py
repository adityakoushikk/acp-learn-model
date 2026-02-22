"""Base splitter: returns train/val/test indices."""
import numpy as np


class BaseSplitter:
    """Split indices into train/val/test. Subclasses implement split()."""

    def split(
        self,
        n_samples: int,
        y: np.ndarray | None = None,
        X: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (train_indices, val_indices, test_indices) each 1d array of ints.
        """
        raise NotImplementedError

    def __call__(
        self,
        n_samples: int,
        y: np.ndarray | None = None,
        X: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.split(n_samples=n_samples, y=y, X=X)
