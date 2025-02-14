from dataclasses import dataclass
from typing import Optional


@dataclass
class RampParams:
    start_value: float = 0.
    end_value: float = 1.
    start_iter: int = -1
    end_iter: int = 0

@dataclass
class IrregularMaskParams:
    max_angle: int = 4
    max_len: int = 60
    max_width: int = 20
    min_times: int = 0
    max_times: int = 10
    # draw_method: str = "LINE"  # [ LINE SQUARE ]
    ramp_params: Optional[RampParams] = None

    invert_proba: float = 0.

@dataclass
class RectangleMaskParams:
    margin: int = 10
    bbox_min_size: int = 30
    bbox_max_size: int = 100
    min_times: int = 0
    max_times: int = 3
    ramp_params: Optional[RampParams] = None

    invert_proba: float = 0.

@dataclass
class SegmentationParams:
    confidence_threshold: float = 0.5
    confidence_threshold: float = 0.5
    rigidness_mode: str = "rigid"
    max_object_area: float = 0.3
    min_mask_area: float = 0.02
    downsample_levels: int = 6
    num_variants_per_mask: int = 4 
    max_mask_intersection: float = 0.5
    max_foreground_coverage: float = 0.5
    max_foreground_intersection: float = 0.5
    max_hidden_area: float = 0.2
    max_scale_change: float = 0.25
    horizontal_flip: bool = True
    max_vertical_shift: float = 0.1
    position_shuffle: bool = True

    invert_proba: float = 0.
    
@dataclass
class SuperResParams:
    min_step: int = 2
    max_step: int = 4
    min_width: int = 1
    max_width: int = 3

    invert_proba: float = 0.

@dataclass
class OutpaintingParams:
    min_padding_percent: float = 0.04
    max_padding_percent: float = 0.49
    left_padding_prob: float = 0.5
    top_padding_prob: float = 0.5
    right_padding_prob: float = 0.5
    bottom_padding_prob: float = 0.5
    is_fixed_randomness: bool = False

    invert_proba: float = 0.
    

@dataclass
class MaskLamaParams:
    irregular_proba: float = 1.
    irregular_params: IrregularMaskParams = IrregularMaskParams(
        max_angle=4,
        max_len=200,
        max_width=100,
        max_times=5,
        min_times=1,
        # draw_method='LINE',
    )

    box_proba: float = 1.
    box_params : RectangleMaskParams = RectangleMaskParams(
       margin=10,
       bbox_min_size=30,
       bbox_max_size=150,
       max_times=4,
       min_times=1, 
    )
    
    segm_proba: float = 0.
    segm_params: SegmentationParams = SegmentationParams()
    
    squares_proba: float = 0.
    squares_params: IrregularMaskParams = IrregularMaskParams()
    
    superres_proba: float = 0.
    superres_params: SuperResParams = SuperResParams()

    outpainting_proba: float = 0.
    outpainting_params: OutpaintingParams = OutpaintingParams()
