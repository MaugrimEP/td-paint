import os
from dataclasses import asdict
from typing import Optional

import numpy as np
import torch
from beartype import beartype as typechecker
from jaxtyping import Float, Int, jaxtyped
from matplotlib import pyplot as plt
from pytorch_lightning.tuner.tuning import Tuner
from torchvision.utils import make_grid
from skimage import color
from skimage.segmentation import mark_boundaries

import wandb
from conf.dataset_params import ValueRange
from conf.main_params import GlobalConfiguration
from conf.model_params import SubLoggingParams

from jaxtyping import Shaped
from einops import repeat
from collections.abc import MutableMapping

from utils.lama_colors import generate_colors
COLORS, _ = generate_colors(151) # 151 - max classes for semantic segmentation

colors = [
    [0, 0, 0],
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [0, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
]

label_colours = dict(zip(range(len(colors)), colors))


@jaxtyped
@typechecker
def mask2rgb(
    mask: Int[torch.Tensor, "b h w"],
    return_tensor: bool = True,
):
    """Mask should not be one-hot-encoded"""
    device = mask.device
    max_class = mask.max()

    mask = mask.cpu().numpy()
    r = mask.copy()
    g = mask.copy()
    b = mask.copy()

    for class_i in range(max_class + 1):
        r[mask == class_i] = label_colours[class_i][0]
        g[mask == class_i] = label_colours[class_i][1]
        b[mask == class_i] = label_colours[class_i][2]

    if len(mask.shape) == 2:
        h, w = mask.shape
        rgb = np.zeros([3, h, w])

        rgb[0, :, :] = r.reshape(h, w) / 255.0
        rgb[1, :, :] = g.reshape(h, w) / 255.0
        rgb[2, :, :] = b.reshape(h, w) / 255.0
    else:
        bs, h, w = mask.shape
        rgb = np.zeros([bs, 3, h, w])

        rgb[:, 0, :, :] = r.reshape(bs, h, w) / 255.0
        rgb[:, 1, :, :] = g.reshape(bs, h, w) / 255.0
        rgb[:, 2, :, :] = b.reshape(bs, h, w) / 255.0

    if return_tensor:
        return torch.Tensor(rgb).to(device)
    return rgb


def is_logging_time(
    logging_params: SubLoggingParams, current_epoch, batch_idx, stage
) -> bool:
    if "train" in stage:
        logging_frequencies = logging_params.frequencies[0]
        log_first = logging_params.log_first[0]
    elif "valid" in stage:
        logging_frequencies = logging_params.frequencies[1]
        log_first = logging_params.log_first[1]
    elif "test" in stage:
        logging_frequencies = logging_params.frequencies[2]
        log_first = logging_params.log_first[2]
    else:
        raise Exception(f"Unknown stage: {stage=}")

    if stage not in logging_params.stages:
        return False

    if logging_params.logging_mode is None:
        return False
    elif logging_params.logging_mode == "epoch":
        return (current_epoch % logging_frequencies == 0) and (
            True if log_first else current_epoch != 0
        )
    elif logging_params.logging_mode == "batch":
        return (batch_idx % logging_frequencies == 0) and (
            True if log_first else batch_idx != 0
        )
    else:
        raise Exception(f"Unknown logging mode: {logging_params.logging_mode=}")


def display_tensor(
    tensor: torch.Tensor, unnormalize: bool = False, dpi: Optional[int] = None, save_name: Optional[str] = None,
):
    """
    Debugging function to display tensor on screen
    """
    if unnormalize:
        tensor = (tensor + 1) / 2
    if len(tensor.shape) == 4:  # there is the batch is the shape -> make a grid
        tensor = make_grid(tensor, padding=20)
    if dpi is not None:
        plt.figure(dpi=dpi)
    plt.imshow(tensor.permute(1, 2, 0).cpu())
    if save_name is not None:
        plt.savefig(save_name)
    plt.show()


def display_mask(tensor: torch.Tensor, dpi: Optional[int] = None, save_name: Optional[str] = None) -> None:
    """
    Debugging function to display mask on screen
    """
    tensor = mask2rgb(tensor)
    display_tensor(tensor=tensor, unnormalize=False, dpi=dpi, save_name=save_name)


def normalize_value_range(
    tensor: torch.Tensor, value_range: ValueRange, clip: bool = False
):
    if value_range == ValueRange.Zero:
        res = tensor
    elif value_range == ValueRange.ZeroUnbound:
        res = tensor
    elif value_range == ValueRange.One:
        res = (tensor + 1) / 2
    elif value_range == ValueRange.OneUnbound:
        res = (tensor + 1) / 2
    else:
        raise Exception(f"Unknown value range: {value_range=}")

    return res if not clip else torch.clamp(res, 0.0, 1.0)


def get_undersample_indices(
    src_length: int,
    nb_samples: int,
    strategy: str = "uniform",
    quad_factor: float = 0.8,
) -> list[int]:
    assert nb_samples > 0
    if nb_samples in [1, 2]:
        indices = [src_length - 1] if nb_samples == 1 else [0, src_length - 1]
        return indices

    if src_length <= nb_samples:
        return [i for i in range(src_length)]

    first = 0
    last = src_length - 1
    src_list = [i for i in range(1, src_length - 1)]
    nb_samples -= 2

    if strategy == "uniform":
        step = len(src_list) // nb_samples
        time_steps = [i * step for i in range(0, nb_samples)]
    elif strategy == "quad_start":
        time_steps = (
            (np.linspace(0, np.sqrt(len(src_list) * quad_factor), nb_samples)) ** 2
        ).astype(int) + 1
        time_steps = [len(src_list) - i for i in time_steps][::-1]
    elif strategy == "quad_end":
        time_steps = (
            (np.linspace(0, np.sqrt(len(src_list) * quad_factor), nb_samples)) ** 2
        ).astype(int) + 1
        time_steps = time_steps.tolist()
    else:
        raise ValueError(f"{strategy=} is not an available discretization method.")

    return time_steps + [last]


def undersample_list(
    src_list: list,
    nb_samples: int,
    strategy: str = "uniform",
    quad_factor: float = 0.8,
    return_indices: bool = False,
) -> list:
    assert nb_samples > 0
    if nb_samples in [1, 2]:
        indices = [len(src_list) - 1] if nb_samples == 1 else [0, len(src_list) - 1]
        res = [src_list[i] for i in indices]
        if return_indices:
            return res, indices
        else:
            return res

    if len(src_list) <= nb_samples:
        return src_list

    res = []
    first = src_list[0]
    last = src_list[-1]
    src_list = src_list[1:-1]
    nb_samples -= 2

    if strategy == "uniform":
        step = len(src_list) // nb_samples
        time_steps = [i * step for i in range(0, nb_samples)]
    elif strategy == "quad_start":
        time_steps = (
            (np.linspace(0, np.sqrt(len(src_list) * quad_factor), nb_samples)) ** 2
        ).astype(int) + 1
        time_steps = [len(src_list) - i for i in time_steps][::-1]
    elif strategy == "quad_end":
        time_steps = (
            (np.linspace(0, np.sqrt(len(src_list) * quad_factor), nb_samples)) ** 2
        ).astype(int) + 1
        time_steps = time_steps.tolist()
    else:
        raise ValueError(f"{strategy=} is not an available discretization method.")

    for i in time_steps:
        res.append(src_list[i])

    res = [first] + res + [last]

    if return_indices:
        return res, [0] + time_steps + [len(src_list) - 1]
    else:
        return res


@jaxtyped
@typechecker
def broadcast_modes_to_pixels(
    datas: list[Float[torch.Tensor, "b _ci h w"]],
    modes: Float[torch.Tensor, "b n_dom"],
) -> Float[torch.Tensor, "b c h w"]:
    n_dom = len(datas)
    mode_per_dom = []
    for dom_i in range(n_dom):
        _, c, h, w = datas[dom_i].shape
        mode_i = modes[:, dom_i]
        mode_per_dom.append(mode_i.view(-1, 1, 1, 1).repeat(1, c, h, w))
    modes = torch.cat(mode_per_dom, dim=1)

    return modes


@jaxtyped
@typechecker
def broadcast_modes_to_pixels_shape(
    b,
    n_dom,
    c,
    h,
    w,
    modes: Float[torch.Tensor, "b n_dom"],
) -> Float[torch.Tensor, "b c h w"]:
    mode_per_dom = []
    for dom_i in range(n_dom):
        mode_i = modes[:, dom_i]
        mode_per_dom.append(mode_i.view(-1, 1, 1, 1).repeat(1, c, h, w))
    modes = torch.cat(mode_per_dom, dim=1)

    return modes


def learning_rate_finder(cfg: GlobalConfiguration, trainer, model, data_module):
    if not cfg.trainer_params.learning_rate_finder_params.auto_lr_find:
        return

    lr_finder_params = cfg.trainer_params.learning_rate_finder_params
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(
        model=model,
        datamodule=data_module,
        **asdict(lr_finder_params.pl_params),
    )
    results = lr_finder.results
    for lr, loss in zip(results["lr"], results["loss"]):
        wandb.log(
            {
                "lr_finder/lr": lr,
                "lr_finder/loss": loss,
            }
        )

    suggestion = lr_finder.suggestion()
    wandb.summary["lr_finder/suggestion"] = suggestion
    if lr_finder_params.pick_suggestion:
        print(f"Pick suggestion {suggestion=}")
        if suggestion is not None:
            new_lr = suggestion
            model.learning_rate = new_lr
    else:
        print(f"Do not pick suggestion {suggestion=}")

    if cfg.trainer_params.learning_rate_finder_params.exit_after_pick:
        wandb.finish()
        quit()


def batch_size_finder(cfg: GlobalConfiguration, trainer, model, data_module):
    if not cfg.trainer_params.batch_size_finder_params.auto_batch_size_finder:
        return

    bs_finder_params = cfg.trainer_params.batch_size_finder_params
    tuner = Tuner(trainer)
    found_batch_size = tuner.scale_batch_size(
        model=model,
        datamodule=data_module,
        **asdict(bs_finder_params.pl_params),
    )
    wandb.summary["batch_size_finder/found_batch_size"] = found_batch_size

    if cfg.trainer_params.batch_size_finder_params.exit_after_pick:
        wandb.finish()
        quit()


def read_list_from_file(file_path: str) -> list[int]:
    with open(file_path, "r") as file:
        content = file.read()
        integer_list = [int(x) for x in content.split(",") if x.strip()]
    return integer_list


def write_list_to_file(file_path: str, integer_list: list[int]):
    if os.path.isfile(file_path):
        raise ValueError(f"File {file_path} already exist.")
    with open(file_path, "w") as file:
        content = ",".join(str(x) for x in integer_list)
        file.write(content)


@jaxtyped
@typechecker
def patch_to_full(
    map: Shaped[torch.Tensor, 'b 1 h w'],
    patch_size: int,
):
    """
    takes a patched map and return the initial map with the values repeated
    """
    res = repeat(map, 'b 1 h w -> b 1 (h ps1) (w ps2)', ps1=patch_size, ps2=patch_size)
    return res


def flatten(dictionary, parent_key='', separator='.'):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


def visualize_mask_and_images(
    images_dict: dict[str, np.ndarray],
    keys: list[str],
    last_without_mask=True,
    rescale_keys=None,
    mask_only_first=None,
    black_mask=False,
    ) -> np.ndarray:
    mask = images_dict['mask'] > 0.5
    result = []
    for i, k in enumerate(keys):
        img = images_dict[k]
        img = np.transpose(img, (1, 2, 0))

        if rescale_keys is not None and k in rescale_keys:
            img = img - img.min()
            img /= img.max() + 1e-5
        if len(img.shape) == 2:
            img = np.expand_dims(img, 2)

        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        elif (img.shape[2] > 3):
            img_classes = img.argmax(2)
            img = color.label2rgb(img_classes, colors=COLORS)

        if mask_only_first:
            need_mark_boundaries = i == 0
        else:
            need_mark_boundaries = i < len(keys) - 1 or not last_without_mask

        if need_mark_boundaries:
            if black_mask:
                img = img * (1 - mask[0][..., None])
            img = mark_boundaries(img,
                                  mask[0],
                                  color=(1., 0., 0.),
                                  outline_color=(1., 1., 1.),
                                  mode='thick')
        result.append(img)
    return np.concatenate(result, axis=1)


def is_there_keys_starting_with(a: dict, keys: list[str]) -> bool:
    """
    Check if there is a key stat with one of the keys in the list
    """
    for k in keys:
        if any(k in key for key in a.keys()):
            return True
    return False


def add_metrics_to_state_dict(
    lightning_ckpt_path: str,
    new_lightning_ckpt_path: str,
    lightning_module,
) -> bool:
    """
    In PL, if I train with the metrics in the LightningModule, I have to add ddp_find_unused_parameters_true
    If I train without, the metrics keys are not in the state dict, then I have to add them for testing if I go
    through the fit function once before testing.
    This function add the metrics keys to the state dict from the current lightning_module

    return if a new ckpt have been issued
    """
    ckpt_dict = torch.load(lightning_ckpt_path)
    state_dict = ckpt_dict['state_dict']

    # if we already have the keys in the state dict, do nothing
    if is_there_keys_starting_with(state_dict, ['train_metrics.', 'valid_metrics.', 'test_metrics.']):
        print("add_metrics_to_state_dict: Metrics keys already in the state dict, return")
        return False

    # we didnt find the keys in the state dict, we add them and save the new state dict
    metrics_dict = {
        'train_metrics': lightning_module.train_metrics.state_dict(),
        'valid_metrics': lightning_module.valid_metrics.state_dict(),
        'test_metrics': lightning_module.test_metrics.state_dict(),
    }
    state_dict.update(metrics_dict)
    ckpt_dict['state_dict'] = state_dict
    torch.save(ckpt_dict, new_lightning_ckpt_path)
    return True
