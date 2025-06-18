import argparse
import glob
import os
import pathlib
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from tqdm import tqdm, trange

from data.utils_imagenet import im2class


def save_image(tensor_img, path):
    """
    tensor_img : [3, H, W], and in [0,1]
    """
    tensor_img = tensor_img.clamp(min=0.0, max=1.0)
    pil = ToPILImage()(tensor_img)
    pil.save(path)


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


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


def from_imagenet_filename(path: str) -> str:
    # eg of filename class_99_id_4225_n01855672_ILSVRC2012_val_00004226.JPEG
    # get filename from path
    filename = os.path.basename(path)
    class_names = filename.split("_")[4]
    return class_names


class DatasetInpainting(torch.utils.data.Dataset):
    def __init__(
        self,
        images_path: str,  # folder where are the images
        image_size: int,   # final image size
        masks_path: str,   # folder where are the masks
        class_cond: bool,  # if it's class cond
        prompt: str,       # a prompt to give to the model
        outdir: str,       # output folder
        blackiskeep: bool, # if blackiskeep, inverse the mask
        split: int,        # indices of the split dataset, 0 indices
        max_split: int,    # max number of splits
        noverwrite: bool,   # if should overwrite the already generated images
        img_extension: str,# extensions for the images
    ):
        self.outdir = outdir
        self.blackiskeep = blackiskeep
        self.prompt = prompt
        self.image_size = image_size

        masks = sorted(
            glob.glob(os.path.join(masks_path, "*_mask.png")) +
            glob.glob(os.path.join(masks_path, "*_mask.jpg")) +
            glob.glob(os.path.join(masks_path, "*_mask.JPEG"))
        )
        images = [
            os.path.join(images_path, os.path.basename(x).replace("_mask.png", f".{img_extension}"))
            for x in masks
        ]
        print(f"Found {len(images)} images"
                f" and {len(masks)} masks in {images_path}")
        assert len(images) == len(masks)

        class_cond_function = from_imagenet_filename
        classes = None
        if class_cond:
            # Assume classes are the first part of the filename,
            # before an underscore.
            print("ImageNet: Compiling classes names")
            class_names = [class_cond_function(path) for path in images]
            sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
            classes = [x for x in class_names]
            print(f"ImageNet: Found {len(sorted_classes):_} classes")

        if max_split != -1:
            dataset_size = len(images)
            indices_splits = [[i for i in range(j, dataset_size, max_split)] for j in range(0, max_split)]
            current_split = indices_splits[split]
            images = [images[i] for i in current_split]
            masks = [masks[i] for i in current_split]
            classes = [classes[i] for i in current_split] if classes is not None else None
            print(f"Split {split+1} of {len(indices_splits)}")

        if noverwrite:  # should not overwrite already generated
            print("Checking if images are already generated")
            already_generated = [os.path.exists(os.path.join(outdir, pathlib.Path('/foo/bar.txt').stem+ '.png')) for x in images]
            print(f"Found {sum(already_generated)} images already generated")
            images = [x for x, y in zip(images, already_generated) if not y]
            masks = [x for x, y in zip(masks, already_generated) if not y]
            if classes is not None:
                classes = [x for x, y in zip(classes, already_generated) if not y]

        self.classes = classes
        self.images = images
        self.masks = masks

        if self.classes is not None and self.prompt != "":
            raise Exception()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        classe = im2class[self.classes[idx]][-1] if self.classes is not None else ""
        if self.prompt != "":
            classe = self.prompt
        classe = classe.replace("_", " ")
        outpath = os.path.join(self.outdir, os.path.split(image_path)[1])

        image = Image.open(image_path).convert("RGB")
        image = center_crop_arr(image, 256)
        image = image.astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image)
        image = image * 2.0 - 1.0
        # resize the torch.tensor
        original_image = image
        image = TF.resize(image, (self.image_size, self.image_size), Image.Resampling.BILINEAR)

        original_mask = Image.open(mask_path).convert('L')
        mask = original_mask.resize((self.image_size,self.image_size), Image.Resampling.NEAREST)

        original_mask = TF.pil_to_tensor(original_mask)
        mask = TF.pil_to_tensor(mask)

        original_mask = (original_mask>0).float()
        mask = (mask>0).float()

        if self.blackiskeep:
            original_mask = 1 - original_mask
            mask = 1 - mask

        return classe, image, mask, os.path.basename(image_path)
