from dataclasses import dataclass, field
from typing import Any, List, Optional

from hydra.core.config_store import ConfigStore

from conf.checkpoint_params import CheckpointParams
from conf.dataset_params import CelebAParams, DatasetParams, ImageNetParams
from conf.distribution_params import DistributionDistanceParams
from conf.evaluation_params import EvaluationParams
from conf.guided_diffusion_params import GuidedDiffusionParams
from conf.model_params import (
    ModelParams,
)
from conf.ressources_params import RessourcesParams
from conf.slurm_params import CfgSlurm
from conf.system_params import SystemParams
from conf.trainer_params import TrainerParams
from conf.wandb_params import WandbParams


@dataclass
class GlobalConfiguration:
    # region default values
    defaults: List[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"dataset_params/data_params": "celeba"},
            # {"dataset_params/data_params": "imagenet"},
        ]
    )

    seed: Optional[int] = 42

    yaml_conf: Optional[list[str]] = (
        "yaml_conf/base.yaml",
        "yaml_conf/celeba1.yaml",
        # "yaml_conf/imagenet.yaml",
        # "yaml_conf/eval.yaml",
    )
    # endregion

    checkpoint_params: CheckpointParams = CheckpointParams()
    dataset_params: DatasetParams = DatasetParams()
    model_params: ModelParams = ModelParams()
    wandb_params: WandbParams = WandbParams()
    cfgSlurm_params: CfgSlurm = CfgSlurm()
    trainer_params: TrainerParams = TrainerParams()
    system_params: SystemParams = SystemParams()
    args: GuidedDiffusionParams = GuidedDiffusionParams()
    ressources_params: RessourcesParams = RessourcesParams()
    distribution_params: DistributionDistanceParams = DistributionDistanceParams()

    evaluation_params: EvaluationParams = EvaluationParams()

# region register config
cs = ConfigStore.instance()

cs.store(name="base_globalConfiguration", node=GlobalConfiguration)

cs.store(group="dataset_params/data_params", name="celeba", node=CelebAParams)
cs.store(group="dataset_params/data_params", name="imagenet", node=ImageNetParams)
# endregion
