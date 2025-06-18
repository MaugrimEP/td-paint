from typing import Dict

import pytorch_lightning as pl
import torch
from beartype import beartype as typechecker
from jaxtyping import Float, jaxtyped
from torchmetrics.image import (KernelInceptionDistance,
                                LearnedPerceptualImagePatchSimilarity,
                                StructuralSimilarityIndexMeasure)

from conf.dataset_params import DatasetParams
from conf.model_params import ModelParams
from utils.Metric.DiversityMetric import DiversityMetric


class MockMetrics:
    """
    When we just want to skip the metrics
    """
    def __init__(self, params: ModelParams, params_data: DatasetParams):
        pass

    def get_dict_generation_cond(
        self, *,
        data: Float[torch.Tensor, 'b 3 h w'],
        prediction: Float[torch.Tensor, 'b 3 h w'],
        mask: Float[torch.Tensor, 'b 1 h w'],
    ) -> Dict:
        return dict()

    @jaxtyped(typechecker=typechecker)
    def get_dict_generation_diversity(
        self,
        batch: Float[torch.Tensor, 'b diversity ci h w'],
        prediction: Float[torch.Tensor, 'b diversity ci h w'],
    ):
        return dict()


class CelebAMetrics(pl.LightningModule):
    def __init__(
        self,
        params: ModelParams,
        params_data: DatasetParams,
    ):
        super().__init__()
        self.params = params

        # GENERATION
        # face metrics
        self.lpips_clamp_face = LearnedPerceptualImagePatchSimilarity(
            net_type='alex',
            reduction='mean',
            normalize=False,  # image are in [-1,1]
        )
        self.ssim_clamp_face = StructuralSimilarityIndexMeasure(data_range=(-1., 1.))
        kids_params = params.logging.kid_params
        self.kid_clamp_face = KernelInceptionDistance(
            feature=kids_params.feature,
            subsets=kids_params.subsets,
            subset_size=kids_params.subset_size,
            degree=kids_params.degree,
            gamma=kids_params.gamma,
            coef=kids_params.coef,
            reset_real_features=kids_params.reset_real_features,
            normalize=kids_params.normalize,
        )
        
        # GENERATION DIVERSITY
        self.diversity = DiversityMetric()

    def get_dict_generation_cond(
        self, *,
        data: Float[torch.Tensor, 'b 3 h w'],
        prediction: Float[torch.Tensor, 'b 3 h w'],
        mask: Float[torch.Tensor, 'b 1 h w'],
    ) -> Dict:
        self.lpips_clamp_face.update(img1=data.clamp(-1, 1), img2=prediction.clamp(-1, 1))
        self.ssim_clamp_face.update(target=data.clamp(-1, 1), preds=prediction.clamp(-1, 1))

        # INFO: ours data are in [-1,1] and they are expected to be in [0,1]
        self.kid_clamp_face.update(
            imgs=prediction.clamp(-1, 1) * 0.5 + 0.5,
            real=False,
        )
        self.kid_clamp_face.update(
            imgs=data.clamp(-1, 1) * 0.5 + 0.5,
            real=True,
        )

        res = dict()
        res |= {f'lpips_face': self.lpips_clamp_face}
        res |= {f'ssim_face': self.ssim_clamp_face}
        res |= {f'kid_face': self.kid_clamp_face}
        return res

    @jaxtyped(typechecker=typechecker)
    def get_dict_generation_diversity(
        self,
        batch: Float[torch.Tensor, 'b diversity ci h w'],
        prediction: Float[torch.Tensor, 'b diversity ci h w'],
    ):
        self.diversity.update(preds=prediction, target=batch)

        return {
            'diversity_face': self.diversity,
        }
