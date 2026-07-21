"""Stratified K-Fold: use one fold as test, split rest into train/val."""
import numpy as np
from sklearn.model_selection import StratifiedKFold

from acp_learn.data.splitters.base_splitter import BaseSplitter


class StratifiedKFoldSplitter(BaseSplitter):
    def __init__(
        self,
        n_splits: int = 5,
        fold_index: int = 0,
        shuffle: bool = True,
        split_seed: int | None = None,
        val_frac: float = 0.1,
    ):
        self.n_splits = n_splits
        self.fold_index = fold_index
        self.shuffle = shuffle
        self.split_seed = split_seed
        if self.split_seed is None:
            self.split_seed = int(np.random.randint(0, np.iinfo(np.int32).max))
        self.val_frac = val_frac

    def split(
        self,
        n_samples: int,
        y: np.ndarray | None = None,
        X: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if y is None:
            y = np.zeros(n_samples)
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.split_seed if self.shuffle else None,
        )
        idx = np.arange(n_samples)
        folds = list(skf.split(idx, y))
        train_val_idx, test_idx = folds[self.fold_index]
        n_val = max(0, int(len(train_val_idx) * self.val_frac))
        rng = np.random.default_rng(self.split_seed)
        perm = rng.permutation(len(train_val_idx))
        train_idx = train_val_idx[perm[n_val:]]
        val_idx = train_val_idx[perm[:n_val]]
        return train_idx, val_idx, test_idx
