"""Lightning module for ACP binary classification with metrics (AUC, MCC, F1, etc.)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import Accuracy, AUROC, MatthewsCorrCoef, F1Score, Precision, Recall


class ACPLitModule(LightningModule):
    def __init__(
        self,
        net: nn.Module,
        optimizer: dict,
        loss: str = "binary_crossentropy",
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net", "class_weights"])
        self.net = net
        pos_weight = None
        if class_weights is not None and class_weights.numel() >= 2:
            pos_weight = (class_weights[1] / class_weights[0]).view(1)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self._accuracy = Accuracy(task="binary")
        self._auc = AUROC(task="binary")
        self._mcc = MatthewsCorrCoef(task="binary", num_classes=2)
        self._f1 = F1Score(task="binary")
        self._precision = Precision(task="binary")
        self._recall = Recall(task="binary")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _step(self, batch: tuple, stage: str):
        x, y = batch
        y_hat = self.forward(x)
        y_float = y.float().unsqueeze(1)
        loss = F.binary_cross_entropy_with_logits(y_hat, y_float.squeeze(-1))
        pred = (torch.sigmoid(y_hat) >= 0.5).long()
        self._accuracy.update(pred, y)
        self._auc.update(torch.sigmoid(y_hat), y)
        self._mcc.update(pred, y)
        self._f1.update(pred, y)
        self._precision.update(pred, y)
        self._recall.update(pred, y)
        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def _epoch_end_metrics(self, stage: str):
        self.log(f"{stage}/accuracy", self._accuracy.compute(), on_epoch=True)
        self.log(f"{stage}/auc", self._auc.compute(), on_epoch=True)
        self.log(f"{stage}/mcc", self._mcc.compute(), on_epoch=True)
        self.log(f"{stage}/f1", self._f1.compute(), on_epoch=True)
        self.log(f"{stage}/precision", self._precision.compute(), on_epoch=True)
        self.log(f"{stage}/recall", self._recall.compute(), on_epoch=True)
        self._accuracy.reset()
        self._auc.reset()
        self._mcc.reset()
        self._f1.reset()
        self._precision.reset()
        self._recall.reset()

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
