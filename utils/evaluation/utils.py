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


def norm_path(path):
    return os.path.basename(path)


@jaxtyped(typechecker=typechecker)
def get_prediction_from_ours_celeba_png(
    params: EvaluationParams,
    idx_in_ds: int,
    path: str,
) -> Float[torch.Tensor, "3 h w"]:
    # test_ema_generate_cond_cond_9785_img.png
    path_to_file = (
        f"{params.folder_predictions}/test_ema_generate_cond_cond_{idx_in_ds}_img.png"
    )
    file = glob.glob(path_to_file)
    assert len(file) == 1, f"Found {len(file)} files for {path_to_file}, should be 1"
    file = file[0]
    # load image to numpy array
    prediction = np.array(Image.open(file).convert("RGB"))
    prediction_tensor = torch.from_numpy(prediction).float() / 255.0
    prediction_tensor = prediction_tensor.permute(2, 0, 1)  # channels first
    prediction_tensor = prediction_tensor * 2 - 1  # from [0,1] to [-1,1]
    return prediction_tensor


@jaxtyped(typechecker=typechecker)
def get_prediction_from_repaint_celeba_png(
    params: EvaluationParams,
    idx_in_ds: int,
    path: str,
) -> Float[torch.Tensor, "3 h w"]:
    # test_van_generate_cond_cond_9785_img.png  # VAN bcs there is no retraining
    path_to_file = (
        f"{params.folder_predictions}/test_van_generate_cond_cond_{idx_in_ds}_img.png"
    )
    file = glob.glob(path_to_file)
    assert len(file) == 1, f"Found {len(file)} files for {path_to_file}, should be 1"
    file = file[0]
    # load image to numpy array
    prediction = np.array(Image.open(file).convert("RGB"))
    prediction_tensor = torch.from_numpy(prediction).float() / 255.0
    prediction_tensor = prediction_tensor.permute(2, 0, 1)  # channels first
    prediction_tensor = prediction_tensor * 2 - 1  # from [0,1] to [-1,1]
    return prediction_tensor


@jaxtyped(typechecker=typechecker)
def get_prediction_from_repaint_imagenet_png(
    params: EvaluationParams,
    idx_in_ds: int,
    path: str,
) -> Float[torch.Tensor, "3 h w"]:
    # test_ema_generate_cond_cond_9_img_classe_332.png -> it's the indice in the ds not in 100
    path_to_file = f"{params.folder_predictions}/test_ema_generate_cond_cond_{idx_in_ds}_img_classe_*.png"
    file = glob.glob(path_to_file)
    assert len(file) == 1, f"Found {len(file)} files for {path_to_file}, should be 1"
    file = file[0]
    # load image to numpy array
    prediction = np.array(Image.open(file).convert("RGB"))
    prediction_tensor = torch.from_numpy(prediction).float() / 255.0
    prediction_tensor = prediction_tensor.permute(2, 0, 1)  # channels first
    prediction_tensor = prediction_tensor * 2 - 1  # from [0,1] to [-1,1]
    return prediction_tensor


@jaxtyped(typechecker=typechecker)
def get_prediction_from_repaint_places_png(
    params: EvaluationParams,
    idx_in_ds: int,
    path: str,
) -> Float[torch.Tensor, "3 h w"]:
    number = norm_path(path).split("_")[1]
    # test_ema_generate_cond_cond_472_img.png -> it's the indice in the ds not in 100
    path_to_file = (
        f"{params.folder_predictions}/test_ema_generate_cond_cond_{number}_img.png"
    )
    file = glob.glob(path_to_file)
    assert len(file) == 1, f"Found {len(file)} files for {path_to_file}, should be 1"
    file = file[0]
    # load image to numpy array
    prediction = np.array(Image.open(file).convert("RGB"))
    prediction_tensor = torch.from_numpy(prediction).float() / 255.0
    prediction_tensor = prediction_tensor.permute(2, 0, 1)  # channels first
    prediction_tensor = prediction_tensor * 2 - 1  # from [0,1] to [-1,1]
    return prediction_tensor


@jaxtyped(typechecker=typechecker)
def get_prediction_from_lama_celeba_png(
    params: EvaluationParams,
    idx_in_ds: int,
    path: str,
) -> Float[torch.Tensor, "3 h w"]:
    path_to_file = os.path.join(
        params.folder_predictions,
        f"{params.dataset}_{idx_in_100}_id_{idx_in_ds}_mask.png",
    )
    # load image to numpy array
    prediction = np.array(Image.open(path_to_file).convert("RGB"))
    prediction_tensor = torch.from_numpy(prediction).float() / 255.0
    prediction_tensor = prediction_tensor.permute(2, 0, 1)  # channels first
    prediction_tensor = prediction_tensor * 2 - 1  # from [0,1] to [-1,1]
    return prediction_tensor


@jaxtyped(typechecker=typechecker)
def get_prediction_from_lama_imagenet_png(
    params: EvaluationParams,
    idx_in_ds: int,
    path: str,
) -> Float[torch.Tensor, "3 h w"]:
    path_to_file = f"{params.folder_predictions}/class_*_id_{idx_in_ds}_*_ILSVRC2012_val_*_mask.png"
    file = glob.glob(path_to_file)
    assert len(file) == 1, f"Found {len(file)} files for {path_to_file}, should be 1"
    file = file[0]
    # load image to numpy array
    prediction = np.array(Image.open(file).convert("RGB"))
    prediction_tensor = torch.from_numpy(prediction).float() / 255.0
    prediction_tensor = prediction_tensor.permute(2, 0, 1)  # channels first
    prediction_tensor = prediction_tensor * 2 - 1  # from [0,1] to [-1,1]
    return prediction_tensor


@jaxtyped(typechecker=typechecker)
def get_prediction_from_lama_places_png(
    params: EvaluationParams,
    idx_in_ds: int,
    path: str,
) -> Float[torch.Tensor, "3 h w"]:
    path_to_file = (
        f"{params.folder_predictions}/id_{idx_in_ds}_Places365_test_*_mask.png"
    )
    file = glob.glob(path_to_file)
    assert len(file) == 1, f"Found {len(file)} files for {path_to_file}, should be 1"
    file = file[0]
    # load image to numpy array
    prediction = np.array(Image.open(file).convert("RGB"))
    prediction_tensor = torch.from_numpy(prediction).float() / 255.0
    prediction_tensor = prediction_tensor.permute(2, 0, 1)  # channels first
    prediction_tensor = prediction_tensor * 2 - 1  # from [0,1] to [-1,1]
    return prediction_tensor


@jaxtyped(typechecker=typechecker)
def get_prediction_from_MCG_celeba_png(
    params: EvaluationParams,
    idx_in_ds: int,
    path: str,
) -> Float[torch.Tensor, "3 h w"]:
    filename = f"{params.dataset}_{idx_in_100}_id_{idx_in_ds}.png_sample_0.png"
    path_to_file = os.path.join(params.folder_predictions, filename)
    # load image to numpy array
    prediction = np.array(Image.open(path_to_file).convert("RGB"))
    prediction_tensor = torch.from_numpy(prediction).float() / 255.0
    prediction_tensor = prediction_tensor.permute(2, 0, 1)  # channels first
    prediction_tensor = prediction_tensor * 2 - 1  # from [0,1] to [-1,1]
    return prediction_tensor


@jaxtyped(typechecker=typechecker)
def get_prediction_from_MCG_imagenet_png(
    params: EvaluationParams,
    idx_in_ds: int,
    path: str,
) -> Float[torch.Tensor, "3 h w"]:
    # thin/class_779_id_3708_n04146614_ILSVRC2012_val_00003709.JPEG_sample_0.png
    path_to_file = f"{params.folder_predictions}/class_*_id_{idx_in_ds}_*_ILSVRC2012_val_*.JPEG_sample_0.png"
    file = glob.glob(path_to_file)
    assert len(file) == 1, f"Found {len(file)} files for {path_to_file}, should be 1"
    file = file[0]
    # load image to numpy array
    prediction = np.array(Image.open(file).convert("RGB"))
    prediction_tensor = torch.from_numpy(prediction).float() / 255.0
    prediction_tensor = prediction_tensor.permute(2, 0, 1)  # channels first
    prediction_tensor = prediction_tensor * 2 - 1  # from [0,1] to [-1,1]
    return prediction_tensor


@jaxtyped(typechecker=typechecker)
def get_prediction_from_MCG_places_png(
    params: EvaluationParams,
    idx_in_ds: int,
    path: str,
) -> Float[torch.Tensor, "3 h w"]:
    # ex64/id_156_Places365_test_00019929.jpg_sample_0.png
    path_to_file = (
        f"{params.folder_predictions}/id_{idx_in_ds}_Places365_test_*.jpg_sample_0.png"
    )
    file = glob.glob(path_to_file)
    assert len(file) == 1, f"Found {len(file)} files for {path_to_file}, should be 1"
    file = file[0]
    # load image to numpy array
    prediction = np.array(Image.open(file).convert("RGB"))
    prediction_tensor = torch.from_numpy(prediction).float() / 255.0
    prediction_tensor = prediction_tensor.permute(2, 0, 1)  # channels first
    prediction_tensor = prediction_tensor * 2 - 1  # from [0,1] to [-1,1]
    return prediction_tensor


@jaxtyped(typechecker=typechecker)
def get_prediction_from_MAT_celeba_png(
    params: EvaluationParams,
    idx_in_ds: int,
    path: str,
) -> Float[torch.Tensor, "3 h w"]:
    # the MAT filename is just the original filename, eg celeba_987_id_10976.png
    filename = f"{params.dataset}_{idx_in_100}_id_{idx_in_ds}.png"
    path_to_file = os.path.join(params.folder_predictions, filename)
    # load image to numpy array
    prediction = np.array(Image.open(path_to_file).convert("RGB"))
    prediction_tensor = torch.from_numpy(prediction).float() / 255.0
    prediction_tensor = prediction_tensor.permute(2, 0, 1)  # channels first
    prediction_tensor = prediction_tensor * 2 - 1  # from [0,1] to [-1,1]
    return prediction_tensor


@jaxtyped(typechecker=typechecker)
def get_prediction_from_MAT_imagenet_png(
    params: EvaluationParams,
    idx_in_ds: int,
    path: str,
) -> Float[torch.Tensor, "3 h w"]:
    # the MAT filename is just the original filename,
    # eg class_999_id_2662_n15075141_ILSVRC2012_val_00002663.JPEG
    path_to_file = (
        f"{params.folder_predictions}/class_*_id_{idx_in_ds}_*_ILSVRC2012_val_*.JPEG"
    )
    file = glob.glob(path_to_file)
    assert len(file) == 1, f"Found {len(file)} files for {path_to_file}, should be 1"
    file = file[0]
    # load image to numpy array
    prediction = np.array(Image.open(file).convert("RGB"))
    prediction_tensor = torch.from_numpy(prediction).float() / 255.0
    prediction_tensor = prediction_tensor.permute(2, 0, 1)  # channels first
    prediction_tensor = prediction_tensor * 2 - 1  # from [0,1] to [-1,1]
    return prediction_tensor


@jaxtyped(typechecker=typechecker)
def get_prediction_from_MAT_places_png(
    params: EvaluationParams,
    idx_in_ds: int,
    path: str,
) -> Float[torch.Tensor, "3 h w"]:
    # id_989_Places365_test_00156144.png
    path_to_file = os.path.join(params.folder_predictions, path_to_file)
    file = glob.glob(path_to_file)
    assert len(file) == 1, f"Found {len(file)} files for {path_to_file}, should be 1"
    file = file[0]
    # load image to numpy array
    prediction = np.array(Image.open(file).convert("RGB"))
    prediction_tensor = torch.from_numpy(prediction).float() / 255.0
    prediction_tensor = prediction_tensor.permute(2, 0, 1)  # channels first
    prediction_tensor = prediction_tensor * 2 - 1  # from [0,1] to [-1,1]
    return prediction_tensor


@jaxtyped(typechecker=typechecker)
def get_prediction_from_LDM_celeba_png(
    params: EvaluationParams,
    idx_in_ds: int,
    path: str,
) -> Float[torch.Tensor, "3 h w"]:
    # the LAMA filename is just the original filename, eg celeba_987_id_10976.png
    filename = f"celebahq_{idx_in_100}_id_{idx_in_ds}.png"
    path_to_file = os.path.join(params.folder_predictions, filename)
    # load image to numpy array
    prediction = np.array(Image.open(path_to_file).convert("RGB"))
    prediction_tensor = torch.from_numpy(prediction).float() / 255.0
    prediction_tensor = prediction_tensor.permute(2, 0, 1)  # channels first
    prediction_tensor = prediction_tensor * 2 - 1  # from [0,1] to [-1,1]
    return prediction_tensor


@jaxtyped(typechecker=typechecker)
def get_prediction_from_LDM_imagenet_png(
    params: EvaluationParams,
    idx_in_ds: int,
    path: str,
) -> Float[torch.Tensor, "3 h w"]:
    # class_0_id_6696_n01440764_ILSVRC2012_val_00006697.JPEG
    path_to_file = (
        f"{params.folder_predictions}/class_*_id_{idx_in_ds}_*_ILSVRC2012_val_*.JPEG"
    )
    file = glob.glob(path_to_file)
    assert len(file) == 1, f"Found {len(file)} files for {path_to_file}, should be 1"
    file = file[0]
    # load image to numpy array
    prediction = np.array(Image.open(file).convert("RGB"))
    prediction_tensor = torch.from_numpy(prediction).float() / 255.0
    prediction_tensor = prediction_tensor.permute(2, 0, 1)  # channels first
    prediction_tensor = prediction_tensor * 2 - 1  # from [0,1] to [-1,1]
    return prediction_tensor


@jaxtyped(typechecker=typechecker)
def get_prediction_from_LDM_places_png(
    params: EvaluationParams,
    idx_in_ds: int,
    path: str,
) -> Float[torch.Tensor, "3 h w"]:
    number = norm_path(path).split("_")[1]
    # id_1000_Places365_test_00158219.jpg
    path_to_file = f"id_{number}_Places365_test_*.jpg"
    path_to_file = os.path.join(params.folder_predictions, path_to_file)
    file = glob.glob(path_to_file)
    assert len(file) == 1, f"Found {len(file)} files for {path_to_file}, should be 1"
    file = file[0]
    # load image to numpy array
    prediction = np.array(Image.open(file).convert("RGB"))
    prediction_tensor = torch.from_numpy(prediction).float() / 255.0
    prediction_tensor = prediction_tensor.permute(2, 0, 1)  # channels first
    prediction_tensor = prediction_tensor * 2 - 1  # from [0,1] to [-1,1]
    return prediction_tensor


@jaxtyped(typechecker=typechecker)
def get_prediction_from_bld_celeba_png(
    params: EvaluationParams,
    idx_in_ds: int,
    path: str,
) -> Float[torch.Tensor, "3 h w"]:
    # celebahq_991_id_11024.png
    path_to_file = f"celebahq_*_id_{idx_in_ds}.png"
    path_to_file = os.path.join(params.folder_predictions, path_to_file)
    file = glob.glob(path_to_file)
    assert len(file) == 1, f"Found {len(file)} files for {path_to_file}, should be 1"
    file = file[0]
    # load image to numpy array
    prediction = np.array(Image.open(file).convert("RGB"))
    prediction_tensor = torch.from_numpy(prediction).float() / 255.0
    prediction_tensor = prediction_tensor.permute(2, 0, 1)  # channels first
    prediction_tensor = prediction_tensor * 2 - 1  # from [0,1] to [-1,1]
    return prediction_tensor


@jaxtyped(typechecker=typechecker)
def get_prediction_from_copaint_celeba_png(
    params: EvaluationParams,
    idx_in_ds: int,
) -> Float[torch.Tensor, "3 h w"]:
    # celebahq_991_id_11024.png.png
    path_to_file = f"celebahq_*_id_{idx_in_ds}.png.png"
    path_to_file = os.path.join(params.folder_predictions, path_to_file)
    file = glob.glob(path_to_file)
    assert len(file) == 1, f"Found {len(file)} files for {path_to_file}, should be 1"
    file = file[0]
    # load image to numpy array
    prediction = np.array(Image.open(file).convert("RGB"))
    prediction_tensor = torch.from_numpy(prediction).float() / 255.0
    prediction_tensor = prediction_tensor.permute(2, 0, 1)  # channels first
    prediction_tensor = prediction_tensor * 2 - 1  # from [0,1] to [-1,1]
    return prediction_tensor


@jaxtyped(typechecker=typechecker)
def get_prediction_from_bld_imagenet_png(
    params: EvaluationParams,
    idx_in_ds: int,
    path: str,
) -> Float[torch.Tensor, "3 h w"]:
    # class_0_id_2137_n01440764_ILSVRC2012_val_00002138.JPEG
    path_to_file = f"class_*_id_{idx_in_ds}_*_ILSVRC2012_val_*.JPEG"
    path_to_file = os.path.join(params.folder_predictions, path_to_file)
    file = glob.glob(path_to_file)
    assert len(file) == 1, f"Found {len(file)} files for {path_to_file}, should be 1"
    file = file[0]
    # load image to numpy array
    prediction = np.array(Image.open(file).convert("RGB"))
    prediction_tensor = torch.from_numpy(prediction).float() / 255.0
    prediction_tensor = prediction_tensor.permute(2, 0, 1)  # channels first
    prediction_tensor = prediction_tensor * 2 - 1  # from [0,1] to [-1,1]
    return prediction_tensor


@jaxtyped(typechecker=typechecker)
def get_prediction_from_copaint_imagenet_png(
    params: EvaluationParams,
    idx_in_ds: int,
    path: str,
) -> Float[torch.Tensor, "3 h w"]:
    # class_0_id_2137_n01440764_ILSVRC2012_val_00002138.JPEG.png
    path_to_file = f"class_*_id_{idx_in_ds}_*_ILSVRC2012_val_*.JPEG.png"
    path_to_file = os.path.join(params.folder_predictions, path_to_file)
    file = glob.glob(path_to_file)
    assert len(file) == 1, f"Found {len(file)} files for {path_to_file}, should be 1"
    file = file[0]
    # load image to numpy array
    prediction = np.array(Image.open(file).convert("RGB"))
    prediction_tensor = torch.from_numpy(prediction).float() / 255.0
    prediction_tensor = prediction_tensor.permute(2, 0, 1)  # channels first
    prediction_tensor = prediction_tensor * 2 - 1  # from [0,1] to [-1,1]
    return prediction_tensor


@jaxtyped(typechecker=typechecker)
def get_prediction_from_bld_places_png(
    params: EvaluationParams,
    idx_in_ds: int,
    path: str,
) -> Float[torch.Tensor, "3 h w"]:
    number = norm_path(path).split("_")[1]
    # id_1000_Places365_test_00158219.jpg
    path_to_file = f"id_{number}_Places365_test_*.jpg"
    path_to_file = os.path.join(params.folder_predictions, path_to_file)
    file = glob.glob(path_to_file)
    assert len(file) == 1, f"Found {len(file)} files for {path_to_file}, should be 1"
    file = file[0]
    # load image to numpy array
    prediction = np.array(Image.open(file).convert("RGB"))
    prediction_tensor = torch.from_numpy(prediction).float() / 255.0
    prediction_tensor = prediction_tensor.permute(2, 0, 1)  # channels first
    prediction_tensor = prediction_tensor * 2 - 1  # from [0,1] to [-1,1]
    return prediction_tensor


@jaxtyped(typechecker=typechecker)
def get_prediction_from_copaint_places_png(
    params: EvaluationParams,
    idx_in_ds: int,
    path: str,
) -> Float[torch.Tensor, "3 h w"]:
    number = norm_path(path).split("_")[1]
    # id_1000_Places365_test_00158219.jpg.png
    path_to_file = f"id_{number}_Places365_test_*.jpg.png"
    path_to_file = os.path.join(params.folder_predictions, path_to_file)
    file = glob.glob(path_to_file)
    assert len(file) == 1, f"Found {len(file)} files for {path_to_file}, should be 1"
    file = file[0]
    print(file)
    # load image to numpy array
    prediction = np.array(Image.open(file).convert("RGB"))
    prediction_tensor = torch.from_numpy(prediction).float() / 255.0
    prediction_tensor = prediction_tensor.permute(2, 0, 1)  # channels first
    prediction_tensor = prediction_tensor * 2 - 1  # from [0,1] to [-1,1]
    return prediction_tensor


@jaxtyped(typechecker=typechecker)
def get_prediction_from(
    params: EvaluationParams,
    idx_in_ds: int,
    path: str,
) -> Float[torch.Tensor, "3 h w"]:
    match params.get_prediction_from:
        case "lama_celeba":
            return get_prediction_from_lama_celeba_png(params, idx_in_ds, path=path)
        case "lama_imagenet":
            return get_prediction_from_lama_imagenet_png(params, idx_in_ds, path=path)
        case "lama_places":
            return get_prediction_from_lama_places_png(params, idx_in_ds, path=path)

        case "MCG_celeba":
            return get_prediction_from_MCG_celeba_png(params, idx_in_ds, path=path)
        case "MCG_imagenet":
            return get_prediction_from_MCG_imagenet_png(params, idx_in_ds, path=path)
        case "MCG_places":
            return get_prediction_from_MCG_places_png(params, idx_in_ds, path=path)

        case "MAT_celeba":
            return get_prediction_from_MAT_celeba_png(params, idx_in_ds, path=path)
        case "MAT_imagenet":
            return get_prediction_from_MAT_imagenet_png(params, idx_in_ds, path=path)
        case "MAT_places":
            return get_prediction_from_MAT_places_png(params, idx_in_ds, path=path)

        case "repaint_celeba":
            return get_prediction_from_repaint_celeba_png(params, idx_in_ds, path=path)
        case "repaint_imagenet":
            return get_prediction_from_repaint_imagenet_png(
                params, idx_in_ds, path=path
            )
        case "repaint_places":
            return get_prediction_from_repaint_places_png(params, idx_in_ds, path=path)

        case "ours_celeba":
            return get_prediction_from_ours_celeba_png(params, idx_in_ds, path=path)
        case "ours_imagenet":
            return get_prediction_from_repaint_imagenet_png(
                params, idx_in_ds, path=path
            )
        case "ours_places":
            return get_prediction_from_repaint_places_png(params, idx_in_ds, path=path)

        case "LDM_celeba":
            return get_prediction_from_LDM_celeba_png(params, idx_in_ds, path=path)
        case "LDM_imagenet":
            return get_prediction_from_LDM_imagenet_png(params, idx_in_ds, path=path)
        case "LDM_places":
            return get_prediction_from_LDM_places_png(params, idx_in_ds, path=path)

        case "copaint_celeba":
            return get_prediction_from_copaint_celeba_png(params, idx_in_ds, path=path)
        case "copaint_imagenet":
            return get_prediction_from_copaint_imagenet_png(
                params, idx_in_ds, path=path
            )
        case "copaint_places":
            return get_prediction_from_copaint_places_png(params, idx_in_ds, path=path)

        case "bld_celeba":
            return get_prediction_from_bld_celeba_png(params, idx_in_ds, path=path)
        case "bld_imagenet":
            return get_prediction_from_bld_imagenet_png(params, idx_in_ds, path=path)
        case "bld_places":
            return get_prediction_from_bld_places_png(params, idx_in_ds, path=path)

        case _:
            raise ValueError(
                f"Unknown get_prediction_from: {params.get_prediction_from}"
            )
