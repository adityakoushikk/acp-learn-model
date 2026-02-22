from typing import Any
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf


@rank_zero_only
def log_hyperparameters(object_dict: dict[str, Any]) -> None:
    cfg = OmegaConf.to_container(object_dict["cfg"])
    model = object_dict["model"]
    trainer = object_dict["trainer"]
    if not trainer.logger:
        return
    hparams = {
        "model": cfg.get("model"),
        "model/params/total": sum(p.numel() for p in model.parameters()),
        "model/params/trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "data": cfg.get("data"),
        "trainer": cfg.get("trainer"),
        "task_name": cfg.get("task_name"),
        "tags": cfg.get("tags"),
        "seed": cfg.get("seed"),
    }
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)
