import torch
import os
from jaxtyping import jaxtyped, Float
from beartype import beartype
import numpy as np
from PIL import Image
import glob

from conf.evaluation_params import EvaluationParams


"""
Idx in 100 is to rename, it's the relative indices once we have exported the images to subsample them
idx in dataset is more often used for our framework where we kept the initial dataset image idx
"""

@beartype
@jaxtyped
def get_prediction_from_lama_pt(
    params: EvaluationParams,
    idx_in_100: int,
    idx_in_ds: int,
) -> Float[torch.Tensor, "3 h w"]:
    path_to_file = os.path.join(params.folder_predictions, f"{params.dataset}_{idx_in_100}_id_{idx_in_ds}_mask.pt")
    prediction = torch.load(path_to_file)  # data should be in [-1,1] and of float32 type and channel first already
    return prediction


@beartype
@jaxtyped
def get_prediction_from_repaint_celeba_png(
    params: EvaluationParams,
    idx_in_100: int,
    idx_in_ds: int,
) -> Float[torch.Tensor, "3 h w"]:
    # test_van_generate_cond_cond_9785_img.png  # VAN bcs there is no retraining
    path_to_file = f"{params.folder_predictions}/test_van_generate_cond_cond_{idx_in_ds}_img.png"
    file = glob.glob(path_to_file)
    assert len(file) == 1, f"Found {len(file)} files for {path_to_file}, should be 1"
    file = file[0]
    # load image to numpy array
    prediction = np.array(Image.open(file).convert('RGB'))
    prediction_tensor = torch.from_numpy(prediction).float() / 255.
    prediction_tensor = prediction_tensor.permute(2, 0, 1)  # channels first
    prediction_tensor = prediction_tensor * 2 - 1  # from [0,1] to [-1,1]
    return prediction_tensor


@beartype
@jaxtyped
def get_prediction_from_repaint_imagenet_png(
    params: EvaluationParams,
    idx_in_100: int,
    idx_in_ds: int,
) -> Float[torch.Tensor, "3 h w"]:
    # test_ema_generate_cond_cond_9_img_classe_332.png -> it's the indice in the ds not in 100
    path_to_file = f"{params.folder_predictions}/test_ema_generate_cond_cond_{idx_in_ds}_img_classe_*.png"
    file = glob.glob(path_to_file)
    assert len(file) == 1, f"Found {len(file)} files for {path_to_file}, should be 1"
    file = file[0]
    # load image to numpy array
    prediction = np.array(Image.open(file).convert('RGB'))
    prediction_tensor = torch.from_numpy(prediction).float() / 255.
    prediction_tensor = prediction_tensor.permute(2, 0, 1)  # channels first
    prediction_tensor = prediction_tensor * 2 - 1  # from [0,1] to [-1,1]
    return prediction_tensor

@beartype
@jaxtyped
def get_prediction_from_lama_celeba_png(
    params: EvaluationParams,
    idx_in_100: int,
    idx_in_ds: int,
) -> Float[torch.Tensor, "3 h w"]:
    path_to_file = os.path.join(params.folder_predictions, f"{params.dataset}_{idx_in_100}_id_{idx_in_ds}_mask.png")
    # load image to numpy array
    prediction = np.array(Image.open(path_to_file).convert('RGB'))
    prediction_tensor = torch.from_numpy(prediction).float() / 255.
    prediction_tensor = prediction_tensor.permute(2, 0, 1)  # channels first
    prediction_tensor = prediction_tensor * 2 - 1  # from [0,1] to [-1,1]
    return prediction_tensor


@beartype
@jaxtyped
def get_prediction_from_lama_imagenet_png(
    params: EvaluationParams,
    idx_in_100: int,
    idx_in_ds: int,
) -> Float[torch.Tensor, "3 h w"]:
    path_to_file = f"{params.folder_predictions}/class_*_id_{idx_in_ds}_*_ILSVRC2012_val_*_mask.png"
    file = glob.glob(path_to_file)
    assert len(file) == 1, f"Found {len(file)} files for {path_to_file}, should be 1"
    file = file[0]
    # load image to numpy array
    prediction = np.array(Image.open(file).convert('RGB'))
    prediction_tensor = torch.from_numpy(prediction).float() / 255.
    prediction_tensor = prediction_tensor.permute(2, 0, 1)  # channels first
    prediction_tensor = prediction_tensor * 2 - 1  # from [0,1] to [-1,1]
    return prediction_tensor


@beartype
@jaxtyped
def get_prediction_from_MCG_png(
    params: EvaluationParams,
    idx_in_100: int,
    idx_in_ds: int,
) -> Float[torch.Tensor, "3 h w"]:
    filename = f"{params.dataset}_{idx_in_100}_id_{idx_in_ds}.png_sample_0.png"
    path_to_file = os.path.join(params.folder_predictions, filename)
    # load image to numpy array
    prediction = np.array(Image.open(path_to_file).convert('RGB'))
    prediction_tensor = torch.from_numpy(prediction).float() / 255.
    prediction_tensor = prediction_tensor.permute(2, 0, 1)  # channels first
    prediction_tensor = prediction_tensor * 2 - 1  # from [0,1] to [-1,1]
    return prediction_tensor


@beartype
@jaxtyped
def get_prediction_from_MCG_imagenet_png(
    params: EvaluationParams,
    idx_in_100: int,
    idx_in_ds: int,
) -> Float[torch.Tensor, "3 h w"]:
    # thin/class_779_id_3708_n04146614_ILSVRC2012_val_00003709.JPEG_sample_0.png
    path_to_file = f"{params.folder_predictions}/class_*_id_{idx_in_ds}_*_ILSVRC2012_val_*.JPEG_sample_0.png"
    file = glob.glob(path_to_file)
    assert len(file) == 1, f"Found {len(file)} files for {path_to_file}, should be 1"
    file = file[0]
    # load image to numpy array
    prediction = np.array(Image.open(file).convert('RGB'))
    prediction_tensor = torch.from_numpy(prediction).float() / 255.
    prediction_tensor = prediction_tensor.permute(2, 0, 1)  # channels first
    prediction_tensor = prediction_tensor * 2 - 1  # from [0,1] to [-1,1]
    return prediction_tensor


@beartype
@jaxtyped
def get_prediction_from_MAT_celeba_png(
    params: EvaluationParams,
    idx_in_100: int,
    idx_in_ds: int,
) -> Float[torch.Tensor, "3 h w"]:
    # the MAT filename is just the original filename, eg celeba_987_id_10976.png
    filename = f"{params.dataset}_{idx_in_100}_id_{idx_in_ds}.png"
    path_to_file = os.path.join(params.folder_predictions, filename)
    # load image to numpy array
    prediction = np.array(Image.open(path_to_file).convert('RGB'))
    prediction_tensor = torch.from_numpy(prediction).float() / 255.
    prediction_tensor = prediction_tensor.permute(2, 0, 1)  # channels first
    prediction_tensor = prediction_tensor * 2 - 1  # from [0,1] to [-1,1]
    return prediction_tensor


@beartype
@jaxtyped
def get_prediction_from_MAT_imagenet_png(
    params: EvaluationParams,
    idx_in_100: int,
    idx_in_ds: int,
) -> Float[torch.Tensor, "3 h w"]:
    # the MAT filename is just the original filename,
    # eg class_999_id_2662_n15075141_ILSVRC2012_val_00002663.JPEG
    path_to_file = f"{params.folder_predictions}/class_*_id_{idx_in_ds}_*_ILSVRC2012_val_*.JPEG"
    file = glob.glob(path_to_file)
    assert len(file) == 1, f"Found {len(file)} files for {path_to_file}, should be 1"
    file = file[0]
    # load image to numpy array
    prediction = np.array(Image.open(file).convert('RGB'))
    prediction_tensor = torch.from_numpy(prediction).float() / 255.
    prediction_tensor = prediction_tensor.permute(2, 0, 1)  # channels first
    prediction_tensor = prediction_tensor * 2 - 1  # from [0,1] to [-1,1]
    return prediction_tensor

@beartype
@jaxtyped
def get_prediction_from(
    params: EvaluationParams,
    idx_in_100: int,
    idx_in_ds: int,
) -> Float[torch.Tensor, "3 h w"]:
    match params.get_prediction_from:
        case "lama_celeba":
            return get_prediction_from_lama_celeba_png(params, idx_in_100, idx_in_ds)
        case "lama_imagenet":
            return get_prediction_from_lama_imagenet_png(params, idx_in_100, idx_in_ds)
        case "MCG":
            return get_prediction_from_MCG_png(params, idx_in_100, idx_in_ds)
        case "MCG_imagenet":
            return get_prediction_from_MCG_imagenet_png(params, idx_in_100, idx_in_ds)
        case "MAT_celeba":
            return get_prediction_from_MAT_celeba_png(params, idx_in_100, idx_in_ds)
        case "MAT_imagenet":
            return get_prediction_from_MAT_imagenet_png(params, idx_in_100, idx_in_ds)
        case "repaint_celeba":
            return get_prediction_from_repaint_celeba_png(params, idx_in_100, idx_in_ds)
        case "repaint_imagenet":
            return get_prediction_from_repaint_imagenet_png(params, idx_in_100, idx_in_ds)
        case "ours_imagenet":
            return get_prediction_from_repaint_imagenet_png(params, idx_in_100, idx_in_ds)
        case _:
            raise ValueError(f"Unknown get_prediction_from: {params.get_prediction_from}")
