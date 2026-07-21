"""Fixed test set: train/val from main data, test from separate indices.
The datamodule will pass precomputed train_val_indices and test_indices
when it has loaded a fixed test set. This splitter just partitions
train_val_indices into train/val."""
import numpy as np

from acp_learn.data.splitters.base_splitter import BaseSplitter


class FixedTestSplitter(BaseSplitter):
    """Use when test set is fixed (e.g. acp-240). Datamodule handles merging;
    this only splits the non-test part into train/val."""

    def __init__(
        self,
        train_frac: float = 0.9,
        val_frac: float = 0.1,
        split_seed: int | None = None,
    ):
        self.train_frac = train_frac
        self.val_frac = val_frac
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
        n_val = max(0, int(n_samples * self.val_frac))
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        test_idx = np.array([], dtype=np.int64)
        return train_idx, val_idx, test_idx

    def split_train_val_only(
        self,
        train_val_indices: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split train_val_indices into train and val. Used by datamodule when test is fixed."""
        n = len(train_val_indices)
        n_val = max(0, int(n * self.val_frac))
        rng = np.random.default_rng(self.split_seed)
        perm = rng.permutation(n)
        val_idx = train_val_indices[perm[:n_val]]
        train_idx = train_val_indices[perm[n_val:]]
        return train_idx, val_idx
