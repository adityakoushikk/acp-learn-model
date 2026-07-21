"""ACP training entrypoint: Hydra + Lightning + WANDB."""
from __future__ import annotations

import os
from pathlib import Path

import hydra
import lightning as L
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

from acp_learn.utils import (
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

# Ensure project root is on path when running as script
_ROOT = Path(__file__).resolve().parents[1]
if _ROOT not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_ROOT))
if os.environ.get("PROJECT_ROOT") is None:
    os.environ["PROJECT_ROOT"] = str(_ROOT.parent)


@task_wrapper
def train(cfg: DictConfig) -> tuple[dict, dict]:
    if cfg.get("seed") is not None:
        L.seed_everything(cfg.seed, workers=True)

    log = __import__("logging").getLogger(__name__)
    log.info("Instantiating datamodule <%s>", cfg.data._target_)
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup("fit")

    # Resolve input_dim for the net from datamodule
    net_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model.net, resolve=True))
    net_cfg.input_dim = datamodule.n_features
    net = hydra.utils.instantiate(net_cfg)
    log.info("Instantiating model <%s>", cfg.model._target_)
    model: LightningModule = hydra.utils.instantiate(
        cfg.model,
        net=net,
        class_weights=datamodule.class_weights,
        _recursive_=False,
    )

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))
    log.info("Instantiating loggers...")
    loggers: list[Logger] = instantiate_loggers(cfg.get("logger"))
    log.info("Instantiating trainer...")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

    object_dict = {"cfg": cfg, "datamodule": datamodule, "model": model, "callbacks": callbacks, "logger": loggers, "trainer": trainer}
    if loggers:
        log_hyperparameters(object_dict)

    if cfg.get("train", True):
        log.info("Starting training")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    metric_dict = dict(trainer.callback_metrics)
    if cfg.get("test", True):
        log.info("Starting testing")
        ckpt_path = getattr(trainer.checkpoint_callback, "best_model_path", None) if trainer.checkpoint_callback else None
        if ckpt_path == "":
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=False if ckpt_path else None)
        metric_dict = {**metric_dict, **trainer.callback_metrics}

    for logger in loggers:
        logger.finalize("success")

    return metric_dict, object_dict


_CONFIG_PATH = str(Path(__file__).resolve().parents[2] / "configs")


@hydra.main(version_base=None, config_path=_CONFIG_PATH, config_name="train.yaml")
def main(cfg: DictConfig):
    if cfg.get("extras", {}).get("print_config"):
        print(OmegaConf.to_yaml(cfg))
    metric_dict, _ = train(cfg)
    return get_metric_value(metric_dict, cfg.get("optimized_metric"))


if __name__ == "__main__":
    main()
