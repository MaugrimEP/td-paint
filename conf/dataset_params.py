from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union

from omegaconf import MISSING, SI

from conf._util import return_factory
from conf.mask_lama_params import MaskLamaParams


@dataclass
class MaskParams:
    mask_root: str = r"gt_keep_mask"
    mask_type: tuple[str] = (
        "ev2li",
        "ex64",
        "genhalf",
        "nn2",
        "thick",
        "thin",
    )
    height: int = SI("${dataset_params.data_params.height}")
    width: int = SI("${dataset_params.data_params.width}")
    return_path: bool = False  # return the path to the mask image for other than lama masks
    
    # proba to use lama mask during training #TODO: deplace that in method params 
    lama_mask_proba: float = 0.
    lama_mask_params: MaskLamaParams = return_factory(MaskLamaParams())


class ValueRange(Enum):
    Zero = "01"
    ZeroUnbound = "01"
    One = "11"
    OneUnbound = "11unbound"


@dataclass
class ImageMaskDatasetParams:
    """
    Used for generate images dataset with masks provided as
    - images in init_image
    - masks provided in mask

    and output the predictions in
    - outdir
    """
    name: str = "imagemask"
    split: int = 0
    max_split: int = -1  # -1 for no split
    batch_size: int = 4
    img_extension: str = "png"
    class_cond: bool = False
    noverwrite: bool = False  # If True, do not remcompute already dones images
    prompt: str = ""

    outdir: str = "predictions/"

    init_image: str = "images/"
    mask: str = "masks/"
    blackiskeep: bool = False

    # not used directly just for intepolation
    image_size: int = 256
    height: int = SI("${dataset_params.data_params.image_size}")
    width: int = SI("${dataset_params.data_params.image_size}")
    channels: int = 3  # only for on domain, for the generation

    value_range: ValueRange = ValueRange.One

    return_indice: bool = True
    return_path: bool = True  # return the img path in the kwargs


# region dataset spec params
@dataclass
class CelebAParams:
    name: str = "celeba"
    root: str = r"/default/path/to/root"
    train_file: str = r"splits/celebahq_train.txt"
    valid_file: str = r"splits/celebahq_valid.txt"
    test_file: str = r"splits/celebahq_test.txt"
    apply_augmentation_on_valid: bool = False
    apply_augmentation_on_test: bool = False
    image_size: int = 256
    height: int = SI("${dataset_params.data_params.image_size}")
    width: int = SI("${dataset_params.data_params.image_size}")
    channels: int = 3  # only for on domain, for the generation

    value_range: ValueRange = ValueRange.One

    random_flip: bool = True
    return_indice: bool = False
    return_path: bool = False  # return the img path in the kwargs


@dataclass
class ImageDatasetParams:
    """
    General image dataset without masks
    """
    name: str = "imagedataset"

    root_train: str = "path/to/train"
    root_valid: str = "path/to/valid"
    root_test: str = "path/to/test"

    # file used to filter the dataset and only keep a subpart 
    valid_file: Optional[str] = None  
    test_file: Optional[str] = None  

    apply_augmentation_on_valid: bool = False
    apply_augmentation_on_test: bool = False
    
    image_size: int = 256
    class_cond: bool = False
    height: int = SI("${dataset_params.data_params.image_size}")
    width: int = SI("${dataset_params.data_params.image_size}")
    channels: int = 3  # only for on domain, for the generation

    value_range: ValueRange = ValueRange.One  # returning data should be in [-1,1]

    random_crop: bool = True  # if True apply random crop, otherwise apply center crop
    random_flip: bool = True

    return_indice: bool = True
    return_class: bool = False
    
    max_images: Optional[int] = None  # if set, we will early stop while searching for recursive images, can be set for debug
    return_path: bool = False  # return the img path in the kwargs


@dataclass
class ImageNetParams:
    name: str = "imagenet"
    root: str = "/defalut/imagenet/path/to/root"

    valid_file: Optional[str] = None  # r"splits/imagenet_100_1.txt"
    test_file: Optional[str] = None  # r"splits/imagenet_100_2.txt"
    apply_augmentation_on_valid: bool = False
    apply_augmentation_on_test: bool = False
    
    image_size: int = 256
    class_cond: bool = True
    height: int = SI("${dataset_params.data_params.image_size}")
    width: int = SI("${dataset_params.data_params.image_size}")
    channels: int = 3  # only for on domain, for the generation

    value_range: ValueRange = ValueRange.One

    random_crop: bool = True  # if True apply random crop, otherwise apply center crop
    random_flip: bool = True

    return_indice: bool = True
    return_class: bool = True
    
    max_images: Optional[int] = None  # if set, we will early stop while searching for recursive images, can be set for debug
    return_path: bool = False  # return the img path in the kwargs
# endregion


@dataclass
class DatasetParams:
    # if set to a file path, fetch idx to keep and remove the others (which is the inverse of the other who remove samples)
    inv_filter_from_file: Optional[str] = None
    """
    "splits/imagenet_5000_idx.txt" -> 5 images per class in the imagenet dataset
    "splits/celebahq_test_100.txt" -> 100 image for celebahq diversity test set
    """

    # if set to a file path, fetch already generated images to remove associated index from the dataset
    filter_from_file: Optional[str] = None
    # if set to a path, fetch already generated images to remove associated index from the dataset
    filter_from_results: Optional[str] = None
    filter_func: Optional[str] = None
    """
    Function used to filter the filename if filter_from_results is not None
    - celebahq_cond
    - imagenet_cond
    """

    data_params: Any = MISSING
    mask_params: MaskParams = return_factory(MaskParams())

    shuffle_val: bool = False
    shuffle_test: bool = False

    drop_last_train: bool = True
    drop_last_valid: bool = False
    drop_last_test: bool = False

    batch_size: int = 64
    batch_size_val: int = SI("${dataset_params.batch_size}")
    batch_size_test: int = SI("${dataset_params.batch_size}")
    use_min_for_batch_size: bool = True  # if drop_last_train is True and len(train)<batch size, set batch size to len(train)
    workers: int = 8
    pin_memory: bool = True

    train_prop: Union[float, int, str] = 0.80
    valid_prop: Union[float, int, str] = 0.10
    test_prop: Union[float, int, str] = 0.10
    file_path: Optional[
        str
    ] = None  # if not None, use this file to order the indices in the dataset before splitting
    proportion_mode: str = "frac"
    """
    how to interpret the proportions
        - frac: [float] values are proportions, eg 0.5 is 50%
        - perc: [int, float] values are percentages, eg 50 is 50%
        - abso: [int] values are absolutes, specify the number of samples for each parts
    """
    
    limit_train_prop: Optional[Union[float, int, str]] = None
    limit_valid_prop: Optional[Union[float, int, str]] = None
    limit_test_prop: Optional[Union[float, int, str]] = None
    limit_proportion_mode: str = "abso"
    """
    how to interpret the proportions
        - frac: [float] values are proportions, eg 0.5 is 50%
        - perc: [int, float] values are percentages, eg 50 is 50%
        - abso: [int] values are absolutes, specify the number of samples for each parts
    """
