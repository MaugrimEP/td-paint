from dataclasses import dataclass
from typing import Optional

from conf.jump_params import JumpParams
from omegaconf import SI


@dataclass
class GuidedDiffusionParams:
    scheduler_jump_params: JumpParams = JumpParams()
    
    network_img_size: Optional[int] = None  # the image size see by the network, not an initial param

    attention_resolutions: tuple[int] = (32, 16, 8)
    channel_mult: Optional[tuple[int]] = None
    class_cond: bool = False
    use_random_class: bool = False  # USE RANDOM CLASS FOR THE GENERATION
    diffusion_steps: int = 1_000
    dropout: float = 0.0
    image_size: int = SI("${dataset_params.data_params.image_size}")
    learn_sigma: bool = True
    # lr_anneal_steps': 0,
    noise_schedule: str = "linear"
    num_channels: int = 256
    num_head_channels: int = 64
    num_heads: int = 4
    num_heads_upsample: int = -1
    num_res_blocks: int = 2
    predict_xstart: bool = False
    resblock_updown: bool = True
    rescale_learned_sigmas: bool = False
    rescale_timesteps: bool = False
    schedule_sampler: str = "uniform"
    timestep_respacing: str = "250"
    use_kl: bool = False
    use_new_attention_order: bool = False
    use_scale_shift_norm: bool = True

    activation_checkpoint: bool = True
    # if should apply activation checkpointing in the UnetMultiTime
    # only works with cache activated

    # classifier params
    use_classifier: bool = False
    classifier_path: str = "/default/classifier/path/256x256_classifier.pt"
    classifier_scale: float = 1.0
    lr_kernel_n_std: int = 2
    classifier_use_fp16: bool = False
    classifier_width: int = 128
    classifier_depth: int = 2
    classifier_attention_resolutions: list[int] = (32, 16, 8)
    classifier_use_scale_shift_norm: bool = True
    classifier_resblock_updown: bool = True
    classifier_pool: str = 'attention'

    use_fp16: bool = False  # if True then use_gd_optimizer must be set to True, as well as classifier_use_fp16
    fp16_scale_growth: float = 1e-3
    # weight_decay': 0.0
    use_gd_optimizer: bool = False
    use_gd_ema: bool = False
    
    ################################################

    model_path: Optional[
        str
    ] = r"/default/pretrained/path/celeba256_250000.pt"

    # generation params
    use_ddim: bool = False
    clip_denoised: bool = True

    # method parameters
    t_mode: str = "one_per_pixel"
    """
    vanilla: one t per domain
    map_vanilla: one t per domain, but map to the same dim than the image
    one_per_pixel: one t per pixel
    """
    t_strategy: str = "random"
    """
    For training only:
    random: during training, sample randomly t
    condition: sample randomly condition pixel which are set to t_clean_value and others get random t
    """
    input_mode: str = "noisy"
    """
    noisy: fuse with noisy condition
    clean: fuse with clean condition
    """
    down_sample_strat: str = "time_map"
    use_cache_strategy: bool = True
    """
    if t_mode is one_per_pixel of map_vanilla:
    time_map: downsample the first input time
    time_emb: downsample the time embedded map
    """
    generation_t_map: str = 't'
    """
    t: will return the scalar t
    t_map: will return the map of t without masking
    t_map_mask: will return the map of t with masking according to the condition if available
    """

    t_clean_value: int = 0
    patch_size_train: list[int] = (1, )
    patch_weight_train: list[int] = (1, )
    patch_size_gen: int = 1
    condition_proba: list[float] = (0.5, )  # list of length 1 for a fixed value, and 2 for a range
    learn_the_condition: bool = False
