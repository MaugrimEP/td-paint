from typing import Union

import torch
from beartype import beartype
from einops import rearrange, repeat
from jaxtyping import Float, jaxtyped
from lpips import lpips
from torch import Tensor
from torchmetrics import Metric


class DiversityMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.loss_fn_alex_sp = lpips.LPIPS(spatial=True)

    @jaxtyped
    @beartype
    def update(
        self,
        preds: Float[Tensor, "b n c h w"],
        target: Union[
            Float[Tensor, "b c h w"],
            Float[Tensor, "b n c h w"],
        ],
    ) -> None:
        """
        Tensor should be normalized in range [-1, 1]
        """
        b, n, c, h, w = preds.shape
        if target.dim() == 4:
            target = repeat(target, "b c h w -> b n c h w", n=n)

        preds, target = rearrange([preds, target], "l b n c h w -> l (b n) c h w")

        lpips_v = self.loss_fn_alex_sp(preds, target)  # [b * n, 1, 256, 256]
        lpips_v = rearrange(lpips_v, "(b n) 1 h w -> b n h w", b=b, n=n)

        lpipses_gl = lpips_v.mean(dim=[2, 3])  # [b, n]

        lpips_gl, _ = lpipses_gl.min(dim=1)  # [b]

        lpips_best_sp, _ = torch.min(lpips_v, dim=1)  # [b, 256, 256]
        lpips_loc = lpips_best_sp.mean(dim=[1, 2])  # [b]
        score = (lpips_gl - lpips_loc) / lpips_gl * 100  # [b]

        self.score += score.sum()
        self.total += b

    def compute(self) -> Tensor:
        return self.score.float() / self.total
