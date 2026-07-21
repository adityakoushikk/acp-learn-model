"""Lightning module for ACP binary classification with metrics (AUC, MCC, F1, etc.)."""
import torch
import torch.nn as nn
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf
from torchmetrics import Accuracy, AUROC, MatthewsCorrCoef, F1Score, MetricCollection, Precision, Recall


class ACPLitModule(LightningModule):
    def __init__(
        self,
        net: nn.Module,
        optimizer: dict,
        loss: str = "binary_crossentropy",
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        if isinstance(optimizer, DictConfig):
            optimizer = OmegaConf.to_container(optimizer, resolve=True)
        self.save_hyperparameters(logger=False, ignore=["net", "class_weights"])
        self.net = net
        pos_weight = None
        if class_weights is not None and class_weights.numel() >= 2:
            pos_weight = (class_weights[1] / class_weights[0]).view(1)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        metrics = MetricCollection({
            "accuracy": Accuracy(task="binary"),
            "auc": AUROC(task="binary"),
            "mcc": MatthewsCorrCoef(task="binary", num_classes=2),
            "f1": F1Score(task="binary"),
            "precision": Precision(task="binary"),
            "recall": Recall(task="binary"),
        })
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _step(self, batch: tuple, stage: str):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y.float())
        probabilities = torch.sigmoid(y_hat)
        getattr(self, f"{stage}_metrics").update(probabilities, y)
        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def _epoch_end_metrics(self, stage: str):
        metrics = getattr(self, f"{stage}_metrics")
        self.log_dict(metrics.compute(), on_epoch=True)
        metrics.reset()

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def on_train_epoch_end(self):
        self._epoch_end_metrics("train")

    def on_validation_epoch_end(self):
        self._epoch_end_metrics("val")

    def on_test_epoch_end(self):
        self._epoch_end_metrics("test")

    def configure_optimizers(self):
        from hydra.utils import instantiate
        return instantiate(self.hparams.optimizer, params=self.parameters())
