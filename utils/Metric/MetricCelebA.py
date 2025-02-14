from typing import Dict

import pytorch_lightning as pl
import torch
from beartype import beartype as typechecker
from jaxtyping import Float, jaxtyped
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity, StructuralSimilarityIndexMeasure

from conf.dataset_params import DatasetParams
from conf.model_params import ModelParams

from utils.Metric.DiversityMetric import DiversityMetric


class MockMetrics:
    """
    When we just want to skip the metrics
    """
    def __init__(self, params: ModelParams, params_data: DatasetParams):
        pass

    @jaxtyped
    def get_dict_generation_cond(
        self, *,
        data: Float[torch.Tensor, 'b 3 h w'],
        prediction: Float[torch.Tensor, 'b 3 h w'],
        mask: Float[torch.Tensor, 'b 1 h w'],
    ) -> Dict:
        return dict()

    @jaxtyped
    @typechecker
    def get_dict_generation_diversity(
        self,
        batch: Float[torch.Tensor, 'b diversity ci h w'],
        prediction: Float[torch.Tensor, 'b diversity ci h w'],
    ):
        return dict()


class CelebAMetrics(pl.LightningModule):
    def __init__(self, params: ModelParams, params_data: DatasetParams):
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
        
        # GENERATION DIVERSITY
        self.diversity = DiversityMetric()

    @jaxtyped
    def get_dict_generation_cond(
        self, *,
        data: Float[torch.Tensor, 'b 3 h w'],
        prediction: Float[torch.Tensor, 'b 3 h w'],
        mask: Float[torch.Tensor, 'b 1 h w'],
    ) -> Dict:
        self.lpips_clamp_face.update(img1=data.clamp(-1, 1), img2=prediction.clamp(-1, 1))
        self.ssim_clamp_face.update(target=data.clamp(-1, 1), preds=prediction.clamp(-1, 1))

        res = dict()
        res |= {f'lpips_face': self.lpips_clamp_face}
        res |= {f'ssim_face': self.ssim_clamp_face}
        return res

    @jaxtyped
    @typechecker
    def get_dict_generation_diversity(
        self,
        batch: Float[torch.Tensor, 'b diversity ci h w'],
        prediction: Float[torch.Tensor, 'b diversity ci h w'],
    ):
        self.diversity.update(preds=prediction, target=batch)

        return {
            'diversity_face': self.diversity,
        }
