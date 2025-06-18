from dataclasses import dataclass, field
from typing import Any, List, Optional

from hydra.core.config_store import ConfigStore

from conf._util import return_factory
from conf.checkpoint_params import CheckpointParams
from conf.dataset_params import (CelebAParams, DatasetParams,
                                 ImageDatasetParams, ImageMaskDatasetParams,
                                 ImageNetParams)
from conf.distribution_params import DistributionDistanceParams
from conf.evaluation_params import EvaluationParams
from conf.guided_diffusion_params import GuidedDiffusionParams
from conf.model_params import ModelParams
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
            {"dataset_params/data_params": "???"},
            # {"dataset_params/data_params": "celeba"},
            # {"dataset_params/data_params": "imagenet"},
            # {"dataset_params/data_params": "imagedataset"},
            # {"dataset_params/data_params": "imagemask"},
        ]
    )

    seed: Optional[int] = 42

    yaml_conf: Optional[list[str]] = (
        "yaml_conf/base.yaml",
        "yaml_conf/celeba1.yaml",
        # "yaml_conf/places.yaml"
        # "yaml_conf/imagemask.yaml"
        # "yaml_conf/imagenet.yaml",
        # "yaml_conf/eval.yaml",
    )
    # endregion

    checkpoint_params: CheckpointParams = return_factory(CheckpointParams())
    dataset_params: DatasetParams = return_factory(DatasetParams())
    model_params: ModelParams = return_factory(ModelParams())
    wandb_params: WandbParams = return_factory(WandbParams())
    cfgSlurm_params: CfgSlurm = return_factory(CfgSlurm())
    trainer_params: TrainerParams = return_factory(TrainerParams())
    system_params: SystemParams = return_factory(SystemParams())
    args: GuidedDiffusionParams = return_factory(GuidedDiffusionParams())
    ressources_params: RessourcesParams = return_factory(RessourcesParams())
    distribution_params: DistributionDistanceParams = return_factory(DistributionDistanceParams())

    evaluation_params: EvaluationParams = return_factory(EvaluationParams())

# region register config
cs = ConfigStore.instance()

cs.store(name="base_globalConfiguration", node=GlobalConfiguration)

cs.store(group="dataset_params/data_params", name="celeba", node=CelebAParams)
cs.store(group="dataset_params/data_params", name="imagenet", node=ImageNetParams)
cs.store(group="dataset_params/data_params", name="imagedataset", node=ImageDatasetParams)
cs.store(group="dataset_params/data_params", name="imagemask", node=ImageMaskDatasetParams)
# endregion
