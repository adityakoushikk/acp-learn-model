"""Lightning DataModule for ACP: load/featureize, split, scale, dataloaders."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset

from acp_learn.data.acp_dataset import ACPDataset
from acp_learn.data.ifeature_runner import run_ifeature_and_merge
def labels_from_ids(
    ids: list[str],
    positive_pattern: str = "seq",
    negative_pattern: str = "TRNEGATIVE",
) -> np.ndarray:
    y = np.zeros(len(ids), dtype=np.int64)
    for i, sid in enumerate(ids):
        if negative_pattern and negative_pattern in str(sid):
            y[i] = 0
        elif positive_pattern and positive_pattern in str(sid):
            y[i] = 1
        else:
            y[i] = 1
    return y


def parse_fasta_ids(fasta_path: str) -> list[str]:
    ids = []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                ids.append(line[1:].split("|")[0].strip())
    return ids


class ACPDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        train_fasta: str,
        test_fasta: str | None = None,
        feature_types: list[str] | None = None,
        ifeature_script: str | None = None,
        ifeature_python: str | None = None,
        features_cache: str | None = None,
        label_positive_pattern: str = "seq",
        label_negative_pattern: str = "TRNEGATIVE",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        imbalance: str = "class_weight",
        scaler: str = "standardize",
        splitter=None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["splitter"])
        self.splitter = splitter
        self.data_dir = Path(data_dir)
        self.train_fasta = Path(train_fasta)
        self.test_fasta = Path(test_fasta) if test_fasta else None
        self.feature_types = feature_types or ["CTDC", "CKSAAGP", "CTDD"]
        self.ifeature_script = ifeature_script or str(self.data_dir.parent / "iFeature" / "iFeature.py")
        self.features_cache = Path(features_cache) if features_cache else None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.imbalance = imbalance
        self.scaler_name = scaler
        self.label_positive_pattern = label_positive_pattern
        self.label_negative_pattern = label_negative_pattern

        self._scaler = None
        self._X_train = None
        self._y_train = None
        self._n_features = None
        self._class_weights = None
        self.data_train = None
        self.data_val = None
        self.data_test = None
        self._full_dataset = None

    def _get_or_compute_features(self, fasta_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
        fasta_path = Path(fasta_path)
        ids = parse_fasta_ids(str(fasta_path))
        cache_dir = self.data_dir / "processed"
        cache_dir.mkdir(parents=True, exist_ok=True)
        types_str = "_".join(self.feature_types)
        cache_path = self.features_cache or (cache_dir / f"features_{types_str}_{fasta_path.stem}.csv")
        cache_path = Path(cache_path)
        if cache_path.exists():
            df = pd.read_csv(cache_path)
        else:
            run_ifeature_and_merge(
                input_fasta=str(fasta_path),
                feature_types=self.feature_types,
                output_csv=str(cache_path),
                ifeature_script=self.ifeature_script,
                python_exe=self.hparams.get("ifeature_python"),
            )
            df = pd.read_csv(cache_path)
        if "Label" in df.columns:
            y = df["Label"].values.astype(np.int64)
            df = df.drop(columns=["Label"])
        else:
            y = labels_from_ids(
                ids[: len(df)],
                self.label_positive_pattern,
                self.label_negative_pattern,
            )
        X = df.values.astype(np.float32)
        return X, y

    def setup(self, stage: str | None = None) -> None:
        X_train, y_train = self._get_or_compute_features(self.train_fasta)
        self._n_features = X_train.shape[1]

        if self.test_fasta and self.test_fasta.exists():
            X_test, y_test = self._get_or_compute_features(self.test_fasta)
            n_train_val = len(X_train)
            X_all = np.vstack([X_train, X_test])
            y_all = np.concatenate([y_train, y_test])
            test_idx = np.arange(n_train_val, len(X_all))
            train_idx, val_idx, _ = self.splitter.split(n_samples=n_train_val, y=y_train, X=X_train)
        else:
            X_all = X_train
            y_all = y_train
            train_idx, val_idx, test_idx = self.splitter.split(
                n_samples=len(X_train), y=y_train, X=X_train
            )

        if self.scaler_name == "standardize":
            self._scaler = StandardScaler()
        elif self.scaler_name == "minmax":
            self._scaler = MinMaxScaler()
        else:
            self._scaler = None
        if self._scaler is not None:
            X_all = self._scaler.fit_transform(X_all)

        self._X_all = X_all.astype(np.float32)
        self._y_all = y_all
        self._full_dataset = ACPDataset(self._X_all, self._y_all)
        self.data_train = Subset(self._full_dataset, train_idx)
        self.data_val = Subset(self._full_dataset, val_idx) if len(val_idx) > 0 else None
        self.data_test = Subset(self._full_dataset, test_idx) if len(test_idx) > 0 else None

        n_pos = (self._y_all == 1).sum()
        n_neg = (self._y_all == 0).sum()
        if self.imbalance == "class_weight" and n_pos > 0 and n_neg > 0:
            w_pos = 1.0 / n_pos
            w_neg = 1.0 / n_neg
            self._class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float32)
            self._class_weights = self._class_weights / self._class_weights.sum() * 2
        else:
            self._class_weights = None

    @property
    def n_features(self) -> int:
        if self._n_features is None:
            raise RuntimeError("Call setup() first")
        return self._n_features

    @property
    def class_weights(self) -> torch.Tensor | None:
        return self._class_weights

    def train_dataloader(self) -> DataLoader:
        sampler = None
        if self.imbalance == "weighted_sampler" and self._class_weights is not None:
            labels = self._y_all[np.array(self.data_train.indices)]
            weights = self._class_weights[labels]
            sampler = WeightedRandomSampler(weights, len(weights))
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader | None:
        if self.data_val is None:
            return None
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader | None:
        if self.data_test is None:
            return None
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
