"""K-Means cluster-based split: assign clusters to train/val/test."""
import numpy as np
from sklearn.cluster import KMeans

from acp_learn.data.splitters.base_splitter import BaseSplitter


class KMeansSplitter(BaseSplitter):
    def __init__(
        self,
        n_clusters: int = 10,
        test_cluster_frac: float = 0.2,
        val_cluster_frac: float = 0.1,
        split_seed: int = 42,
    ):
        self.n_clusters = n_clusters
        self.test_cluster_frac = test_cluster_frac
        self.val_cluster_frac = val_cluster_frac
        self.split_seed = split_seed

    def split(
        self,
        n_samples: int,
        y: np.ndarray | None = None,
        X: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if X is None:
            raise ValueError("KMeansSplitter requires X")
        n_clusters = min(self.n_clusters, n_samples, X.shape[0])
        km = KMeans(n_clusters=n_clusters, random_state=self.split_seed, n_init=10)
        labels = km.fit_predict(X)
        rng = np.random.default_rng(self.split_seed)
        cluster_ids = np.arange(n_clusters)
        rng.shuffle(cluster_ids)
        n_test = max(1, int(n_clusters * self.test_cluster_frac))
        n_val = max(0, int(n_clusters * self.val_cluster_frac))
        test_clusters = set(cluster_ids[:n_test])
        val_clusters = set(cluster_ids[n_test : n_test + n_val])
        train_clusters = set(cluster_ids[n_test + n_val :])
        idx = np.arange(n_samples)
        train_idx = idx[np.isin(labels, list(train_clusters))]
        val_idx = idx[np.isin(labels, list(val_clusters))]
        test_idx = idx[np.isin(labels, list(test_clusters))]
        if len(test_idx) == 0:
            test_idx = idx[-1:]
        return train_idx, val_idx, test_idx
