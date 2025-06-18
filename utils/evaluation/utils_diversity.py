import glob
import os

import numpy as np
import torch
from beartype import beartype as typechecker
from jaxtyping import Float, jaxtyped
from PIL import Image

from conf.evaluation_params import EvaluationParams

"""
Idx in 100 is to rename, it's the relative indices once we have exported the images to subsample them
idx in dataset is more often used for our framework where we kept the initial dataset image idx
"""

@jaxtyped(typechecker=typechecker)
def get_prediction_from_lama_pt(
    params: EvaluationParams,
    idx_in_100: int,
    idx_in_ds: int,
    diversity_index: int,
) -> Float[torch.Tensor, "3 h w"]:
    path_to_file = os.path.join(params.folder_predictions, f"{params.dataset}_{idx_in_100}_id_{idx_in_ds}_mask.pt")
    prediction = torch.load(path_to_file)  # data should be in [-1,1] and of float32 type and channel first already
    return prediction


@jaxtyped(typechecker=typechecker)
def get_prediction_from_ours_celeba_png(
    params: EvaluationParams,
    idx_in_100: int,
    idx_in_ds: int,
    diversity_index: int,
) -> Float[torch.Tensor, "3 h w"]:
    # test_van_generate_diversity_div_9903_8_img.png  # VAN bcs there is no retraining
    path_to_file = f"{params.folder_predictions}/test_ema_generate_diversity_div_{idx_in_ds}_{diversity_index}_img.png"
    file = glob.glob(path_to_file)
    assert len(file) == 1, f"Found {len(file)} files for {path_to_file}, should be 1"
    file = file[0]
    # load image to numpy array
    prediction = np.array(Image.open(file).convert('RGB'))
    prediction_tensor = torch.from_numpy(prediction).float() / 255.
    prediction_tensor = prediction_tensor.permute(2, 0, 1)  # channels first
    prediction_tensor = prediction_tensor * 2 - 1  # from [0,1] to [-1,1]
    return prediction_tensor


@jaxtyped(typechecker=typechecker)
def get_prediction_from_ldm_celeba_png(
    params: EvaluationParams,
    idx_in_100: int,
    idx_in_ds: int,
    diversity_index: int,
) -> Float[torch.Tensor, "3 h w"]:
    # celebahq_983_id_10919_div_0.png
    path_to_file = f"{params.folder_predictions}/celebahq_*_id_{idx_in_ds}_div_{diversity_index}.png"
    file = glob.glob(path_to_file)
    assert len(file) == 1, f"Found {len(file)} files for {path_to_file}, should be 1"
    file = file[0]
    # load image to numpy array
    prediction = np.array(Image.open(file).convert('RGB'))
    prediction_tensor = torch.from_numpy(prediction).float() / 255.
    prediction_tensor = prediction_tensor.permute(2, 0, 1)  # channels first
    prediction_tensor = prediction_tensor * 2 - 1  # from [0,1] to [-1,1]
    return prediction_tensor


@jaxtyped(typechecker=typechecker)
def get_prediction_from_repaint_celeba_png(
    params: EvaluationParams,
    idx_in_100: int,
    idx_in_ds: int,
    diversity_index: int,
) -> Float[torch.Tensor, "3 h w"]:
    # test_van_generate_diversity_div_9903_8_img.png  # VAN bcs there is no retraining
    path_to_file = f"{params.folder_predictions}/test_van_generate_diversity_div_{idx_in_ds}_{diversity_index}_img.png"
    file = glob.glob(path_to_file)
    assert len(file) == 1, f"Found {len(file)} files for {path_to_file}, should be 1"
    file = file[0]
    # load image to numpy array
    prediction = np.array(Image.open(file).convert('RGB'))
    prediction_tensor = torch.from_numpy(prediction).float() / 255.
    prediction_tensor = prediction_tensor.permute(2, 0, 1)  # channels first
    prediction_tensor = prediction_tensor * 2 - 1  # from [0,1] to [-1,1]
    return prediction_tensor


@jaxtyped(typechecker=typechecker)
def get_prediction_from_repaint_imagenet_png(
    params: EvaluationParams,
    idx_in_100: int,
    idx_in_ds: int,
    diversity_index: int,
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

@jaxtyped(typechecker=typechecker)
def get_prediction_from_lama_celeba_png(
    params: EvaluationParams,
    idx_in_100: int,
    idx_in_ds: int,
    diversity_index: int,
) -> Float[torch.Tensor, "3 h w"]:
    path_to_file = os.path.join(params.folder_predictions, f"{params.dataset}_{idx_in_100}_id_{idx_in_ds}_mask.png")
    # load image to numpy array
    prediction = np.array(Image.open(path_to_file).convert('RGB'))
    prediction_tensor = torch.from_numpy(prediction).float() / 255.
    prediction_tensor = prediction_tensor.permute(2, 0, 1)  # channels first
    prediction_tensor = prediction_tensor * 2 - 1  # from [0,1] to [-1,1]
    return prediction_tensor


@jaxtyped(typechecker=typechecker)
def get_prediction_from_lama_imagenet_png(
    params: EvaluationParams,
    idx_in_100: int,
    idx_in_ds: int,
    diversity_index: int,
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


@jaxtyped(typechecker=typechecker)
def get_prediction_from_MCG_png(
    params: EvaluationParams,
    idx_in_100: int,
    idx_in_ds: int,
    diversity_index: int,
) -> Float[torch.Tensor, "3 h w"]:
    # celebahq_983_id_10919.png_sample_9.png
    path_to_file = f"{params.folder_predictions}/celebahq_*_id_{idx_in_ds}.png_sample_{diversity_index}.png"
    file = glob.glob(path_to_file)
    assert len(file) == 1, f"Found {len(file)} files for {path_to_file}, should be 1"
    file = file[0]
    # load image to numpy array
    prediction = np.array(Image.open(file).convert('RGB'))
    prediction_tensor = torch.from_numpy(prediction).float() / 255.
    prediction_tensor = prediction_tensor.permute(2, 0, 1)  # channels first
    prediction_tensor = prediction_tensor * 2 - 1  # from [0,1] to [-1,1]
    return prediction_tensor


@jaxtyped(typechecker=typechecker)
def get_prediction_from_MCG_imagenet_png(
    params: EvaluationParams,
    idx_in_100: int,
    idx_in_ds: int,
    diversity_index: int,
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


@jaxtyped(typechecker=typechecker)
def get_prediction_from_MAT_celeba_png(
    params: EvaluationParams,
    idx_in_100: int,
    idx_in_ds: int,
    diversity_index: int,
) -> Float[torch.Tensor, "3 h w"]:
    # the MAT filename is just the original filename, eg celebahq_942_id_10547_7.png
    path_to_file = f"{params.folder_predictions}/celebahq_*_id_{idx_in_ds}_{diversity_index}.png"
    file = glob.glob(path_to_file)
    assert len(file) == 1, f"Found {len(file)} files for {path_to_file}, should be 1"
    file = file[0]
    # load image to numpy array
    prediction = np.array(Image.open(file).convert('RGB'))
    prediction_tensor = torch.from_numpy(prediction).float() / 255.
    prediction_tensor = prediction_tensor.permute(2, 0, 1)  # channels first
    prediction_tensor = prediction_tensor * 2 - 1  # from [0,1] to [-1,1]
    return prediction_tensor


@jaxtyped(typechecker=typechecker)
def get_prediction_from_MAT_imagenet_png(
    params: EvaluationParams,
    idx_in_100: int,
    idx_in_ds: int,
    diversity_index: int,
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


@jaxtyped(typechecker=typechecker)
def get_prediction_from_copaint_celeba_png(
    params: EvaluationParams,
    idx_in_100: int,
    idx_in_ds: int,
    diversity_index: int,
) -> Float[torch.Tensor, "3 h w"]:
    # celebahq_942_id_10547.png_div_3.png
    path_to_file = f"celebahq_*_id_{idx_in_ds}.png_div_{diversity_index}.png"
    path_to_file = os.path.join(params.folder_predictions, path_to_file)
    file = glob.glob(path_to_file)
    assert len(file) == 1, f"Found {len(file)} files for {path_to_file}, should be 1"
    file = file[0]
    # load image to numpy array
    prediction = np.array(Image.open(file).convert('RGB'))
    prediction_tensor = torch.from_numpy(prediction).float() / 255.
    prediction_tensor = prediction_tensor.permute(2, 0, 1)  # channels first
    prediction_tensor = prediction_tensor * 2 - 1  # from [0,1] to [-1,1]
    return prediction_tensor


@jaxtyped(typechecker=typechecker)
def get_prediction_from(
    params: EvaluationParams,
    idx_in_100: int,
    idx_in_ds: int,
    idx_diversity: int,
) -> Float[torch.Tensor, "3 h w"]:
    match params.get_prediction_from:
        case "lama":
            return get_prediction_from_lama_png(params, idx_in_100, idx_in_ds, idx_diversity)
        case "lama_imagenet":
            return get_prediction_from_lama_imagenet_png(params, idx_in_100, idx_in_ds, idx_diversity)

        case "MCG_celeba":
            return get_prediction_from_MCG_png(params, idx_in_100, idx_in_ds, idx_diversity)
        case "MCG_imagenet":
            return get_prediction_from_MCG_imagenet_png(params, idx_in_100, idx_in_ds, idx_diversity)

        case "MAT_celeba":
            return get_prediction_from_MAT_celeba_png(params, idx_in_100, idx_in_ds, idx_diversity)
        case "MAT_imagenet":
            return get_prediction_from_MAT_imagenet_png(params, idx_in_100, idx_in_ds, idx_diversity)

        case "repaint_celeba":
            return get_prediction_from_repaint_celeba_png(params, idx_in_100, idx_in_ds, idx_diversity)
        case "repaint_imagenet":
            return get_prediction_from_repaint_imagenet_png(params, idx_in_100, idx_in_ds, idx_diversity)

        case "vanilla_celeba":
            return get_prediction_from_repaint_celeba_png(params, idx_in_100, idx_in_ds, idx_diversity)
        case "vanilla_imagenet":
            return get_prediction_from_repaint_imagenet_png(params, idx_in_100, idx_in_ds, idx_diversity)
        
        case "ours_celeba":
            return get_prediction_from_ours_celeba_png(params, idx_in_100, idx_in_ds, idx_diversity)
        case "ours_imagenet":
            return get_prediction_from_ours_imagenet_png(params, idx_in_100, idx_in_ds, idx_diversity)

        case "ldm_celeba":
            return get_prediction_from_ldm_celeba_png(params, idx_in_100, idx_in_ds, idx_diversity)
        case "ldm_imagenet":
            return get_prediction_from_ldm_imagenet_png(params, idx_in_100, idx_in_ds, idx_diversity)

        case "copaint_celeba":
            return get_prediction_from_copaint_celeba_png(params, idx_in_100, idx_in_ds, idx_diversity)

        case _:
            raise ValueError(f"Unknown get_prediction_from: {params.get_prediction_from}")
