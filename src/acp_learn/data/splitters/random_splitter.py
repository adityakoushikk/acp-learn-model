"""Random train/val/test split."""
import numpy as np

from acp_learn.data.splitters.base_splitter import BaseSplitter


class RandomSplitter(BaseSplitter):
    def __init__(
        self,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        split_seed: int | None = None,
    ):
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.split_seed = split_seed
        if self.split_seed is None:
            self.split_seed = int(np.random.randint(0, np.iinfo(np.int32).max))

    def split(
        self,
        n_samples: int,
        y: np.ndarray | None = None,
        X: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.split_seed)
        idx = np.arange(n_samples)
        rng.shuffle(idx)
        n_test = max(1, int(n_samples * self.test_frac))
        n_val = max(0, int(n_samples * self.val_frac))
        n_train = n_samples - n_test - n_val
        train_idx = idx[:n_train]
        val_idx = idx[n_train : n_train + n_val]
        test_idx = idx[n_train + n_val :]
        return train_idx, val_idx, test_idx
