from dataclasses import dataclass
from lightning_utilities.core.rank_zero import rank_zero_only
from typing import Optional

from pytorch_lightning.loggers import WandbLogger

import wandb


@dataclass
class WandbParams:
    project: str = "ProjectName"
    group: str = ""  # the dataset
    name: Optional[str] = None
    notes: Optional[str] = None
    mode: str = "offline"  # [ offline | online | disabled ]
    tags: tuple[str, ...] = ()
    resume: str = "never"  # [ never | must ]
    id: Optional[str] = None


@rank_zero_only
def get_wandb(params: WandbParams, global_dict: dict, additional_conf: Optional[dict] = None):
    return _get_wandb(
        constructor=wandb.init,
        params=params,
        global_dict=global_dict,
        additional_conf=additional_conf,
    )


def get_wandb_logger(
    params: WandbParams, global_dict: dict, additional_conf: Optional[dict] = None
):
    return _get_wandb(
        constructor=WandbLogger,
        params=params,
        global_dict=global_dict,
        additional_conf=additional_conf,
    )


def _get_wandb(
    constructor, params: WandbParams, global_dict: dict, additional_conf: Optional[dict] = None
):
    if additional_conf is None:
        additional_conf = dict()

    to_save_conf = global_dict | additional_conf
    run = constructor(
        project=params.project,
        group=params.group,
        name=params.name,
        notes=params.notes,
        mode=params.mode,
        tags=params.tags,
        resume=params.resume,
        id=params.id,
        config=to_save_conf,
    )
    return run
