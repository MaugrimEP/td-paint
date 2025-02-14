import torch
from jaxtyping import Float, Int
from typing import Optional
from src.Backbones.guided_diffusion.nn import timestep_embedding
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as checkpoint_torch


class VanillaStrategyTime:
    """
    Downsample the time map before input
    """

    def __init__(
        self,
        *,
        model_channels: int,
        time_embedding_module: nn.Module,
        label_emb: Optional[nn.Module],
        num_classes: Optional[int] = None,
        downsample_mode: str = "bilinear",
    ) -> None:
        self.model_channels = model_channels
        self.time_embedding_module = time_embedding_module
        self.label_emb = label_emb
        self.num_classes = num_classes
        self.downsample_mode = downsample_mode

    def get_time_embedding(
        self,
        *,
        timesteps: Float[torch.Tensor, "b 1 h w"],
        initial_size: int,
        target_size: int,
        h_dtype: torch.dtype,
        y: Optional[Int[torch.Tensor, "b 1"]] = None,
        mask: Optional[Int[torch.Tensor, "b 1 h w"]] = None,
        **kwargs,
    ):
        if initial_size != target_size:
            timesteps = F.interpolate(
                timesteps.type(h_dtype), size=target_size, mode=self.downsample_mode
            ).type(timesteps.dtype)

        t_shape = timesteps.shape

        emb_flatten = self.time_embedding_module(
            timestep_embedding(timesteps.flatten(), self.model_channels)
        )  # [M emb_size]
        emb = emb_flatten.reshape(t_shape + emb_flatten.shape[1:])  # [N 1 H W emb_size]

        label_emb = None
        if self.num_classes is not None:
            assert y is not None, "y must be provided if num_classes is not None"
            label_emb = self.label_emb(y)  # shape is [2, 1024]
            label_emb = label_emb[:, None, None, None, :]
            emb = emb + label_emb

        return emb

    def reset_cache(self):
        pass


class CacheStrategyTime:
    def __init__(
        self,
        strategy,
        activation_checkpoint: bool,
    ) -> None:
        self.cache = dict()
        self.strategy = strategy
        self.activation_checkpoint = activation_checkpoint

    def reset_cache(self):
        self.cache = dict()

    def get_time_embedding(
        self,
        *,
        timesteps: Float[torch.Tensor, "b 1 h w"],
        initial_size: int,
        target_size: int,
        h_dtype: torch.dtype,
        y: Optional[Int[torch.Tensor, "b 1"]] = None,
        mask: Optional[Int[torch.Tensor, "b 1 h w"]] = None,
        t_clean: Optional[Int[torch.Tensor, "1"]] = None,
        ts: Optional[Int[torch.Tensor, "b"]] = None,
        **kwargs,
    ):
        if target_size in self.cache:
            emb = self.cache[target_size]
            return emb

        if self.activation_checkpoint:
            emb = checkpoint_torch(
                self.strategy.get_time_embedding,
                timesteps=timesteps,
                initial_size=initial_size,
                y=y,
                mask=mask,
                h_dtype=h_dtype,
                target_size=target_size,
                **kwargs,
                use_reentrant=False,
            )
        else:
            emb = self.strategy.get_time_embedding(
                timesteps=timesteps,
                initial_size=initial_size,
                y=y,
                mask=mask,
                h_dtype=h_dtype,
                target_size=target_size,
                **kwargs,
            )

        self.cache[target_size] = emb
        return emb


def get_t_strategy(down_sample_strat: str):
    if down_sample_strat == "time_map":
        strategy = VanillaStrategyTime
    else:
        raise ValueError(f"Unknown strategy {down_sample_strat}")

    return strategy
