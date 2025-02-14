from dataclasses import dataclass
from typing import Optional

from omegaconf import SI

from conf.dataset_params import ValueRange


@dataclass
class MetricsParams:
    no_metrics: bool = False  # if set to True, do not use Metrics but the Mock Object (for ddp training only), to remove for testing
    name: str = "celeba"
    metrics_logging_stage: tuple[str] = ("train", "valid", "test")
    metrics_logging_freq: tuple[int] = (1, 1, 1)


@dataclass
class LearningRateWarmUp:
    use_scheduler: bool = True
    start_factor: float = 0.3
    end_factor: float = 1.0
    total_iters: int = 5
    last_epoch: int = -1
    verbose: bool = True


@dataclass
class CosinusParams:
    use_scheduler: bool = False
    T_max: int = SI("${trainer_params.max_epochs}")
    eta_min: float = 0.0
    last_epoch: int = -1
    verbose: bool = True


@dataclass
class EMAParams:
    use: bool = True
    decay: float = 0.9999
    validate_original_weights: bool = False  # If True, the EMA Callback will not swap EMA params during the validation steps
    # If False, the EMA Callback will swap EMA params during the validation steps
    every_n_steps: int = 1
    cpu_offload: bool = False

    #######
    # Not Ema constructor parameters
    perform_double_validation: bool = False  # If should perform one validation step over the original weights and one over the ema weights


@dataclass
class OptimizersParams:
    learning_rate: float = 2e-5
    optimizer: str = "adamw"
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    momentum: float = 0.9

    learning_rate_warmup: LearningRateWarmUp = LearningRateWarmUp()
    cosinus_params: CosinusParams = CosinusParams()
    ema: EMAParams = EMAParams()

    max_epochs: int = SI("${trainer_params.max_epochs}")
    max_steps: int = SI("${trainer_params.max_steps}")


@dataclass
class BackboneParams:
    name: str = "celeba_one_t"
    """
    celeba_one_t
    celeba_multi_t
    imagenet_one_t
    celeba_multi_t
    """


@dataclass
class SubLoggingParams:
    logging_mode: Optional[str] = "epoch"  # [ epoch | batch ]
    stages: tuple[str, ...] = ("train", "valid", "test")  # in which stage to log
    frequencies: tuple[int, int, int] = (1, 1, 1)
    log_first: tuple[bool, bool, bool] = (
        True,
        True,
        True,
    )  # whether to log the first batch/epoch
    max_quantity: int = 5
    early_leave: bool = True  # If True, leave once the needed amount of logged image is reached
    save_path: str = "_results/"
    save_mask: bool = False
    save_image_to_disk_stage: tuple[str, ...] = ("test",)
    save_pt_to_disk_stage: tuple[str, ...] = tuple()

@dataclass
class SubLoggingParamsDiversity(SubLoggingParams):
    variation_quantity: Optional[int] = None  # if not None, batch size
    generate_all_in_batch: bool = False  # if False, only generate one random image from the batch


@dataclass
class LoggingParams:
    name: str = "celeba"

    # Logging step params
    log_steps: SubLoggingParams = SubLoggingParams()

    # Logging generate params
    log_generate_uncond: SubLoggingParams = SubLoggingParams()
    log_generate_cond: SubLoggingParams = SubLoggingParams()
    log_generate_diversity: SubLoggingParamsDiversity = SubLoggingParamsDiversity()
    combine_sample_with_mask: bool = True  # if should combine the final single output with the GT and mask

    # Generation Logging parameters
    time_step_in_process: int = (
        10  # number of time steps (from generation denoising process) logged in process
    )
    strategy: str = "quad_end"
    """
        uniform   : uniform time steps
        quad_start: quadratic time steps: more early gen
        quad_end  : quadratic time steps: more end gen
    """
    quad_factor: float = 0.8
    value_range: ValueRange = SI("${dataset_params.data_params.value_range}")


@dataclass
class ModelParams:
    name: str = "diffusion"
    backbone: BackboneParams = BackboneParams()
    logging: LoggingParams = LoggingParams()
    metrics: MetricsParams = MetricsParams()
    optimizer: OptimizersParams = OptimizersParams()
