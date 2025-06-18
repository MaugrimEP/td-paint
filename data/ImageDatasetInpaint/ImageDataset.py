import os
import os.path as osp
import random
from typing import Optional

import blobfile as bf
import numpy as np
import torch
import torch.utils.data as data
import tqdm
from beartype import beartype as typechecker
from jaxtyping import Float, Int, jaxtyped
from PIL import Image

from conf.dataset_params import ImageDatasetParams
from data.ImageDatasetInpaint.utils import center_crop_arr, random_crop_arr
from data.UtilsDataset import CustomDataModule
from utils.utils import display_mask, display_tensor


def _list_image_files_recursively(
    data_dir, log_tqdm: bool = False, early_exit: Optional[int] = None
):
    results = []
    for_range = sorted(bf.listdir(data_dir))
    if log_tqdm:
        print("Listing files in", data_dir)
        for_range = tqdm.tqdm(for_range)
    for entry in for_range:
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
        if early_exit is not None and len(results) >= early_exit:
            return results
    return results


class ImageDataset(data.Dataset):
    def __init__(
        self,
        params: ImageDatasetParams,
        split: str,
    ):
        super().__init__()
        assert split in ['train', 'valid', 'test'], f'Split should be train, valid or test, got {split=}'
        self.params = params
        self.split = split
        self.root = {'train': params.root_train, 'valid': params.root_valid, 'test': params.root_test}[split]

        all_files = _list_image_files_recursively(self.root, log_tqdm=True, early_exit=params.max_images)
        print(f"ImageDataset: Found {len(all_files):_} images")
        # sort files
        print("ImageDataset: Sorting files according to name")
        all_files = sorted(all_files)
        print("ImageDataset: Done sorting files")
        
        file = {
            'train': None,  # no parameter for train
            'valid':params.valid_file,
            'test': params.test_file,
        }[split]
        if file is not None:
            with open(file, 'r') as f:
                kept_files = [file.strip() for file in f.readlines()]
            # filer the all_files list according to kept_files
            print(f"Filtering ImageDataset {split=}, {len(kept_files)=}")
            filtered_files = []
            for file in all_files:
                file_norm = os.path.normpath(file)
                splitted_path = file_norm.split(os.sep)
                classname = splitted_path[-2]
                fileid = splitted_path[-1].split('.')[0].split('_')[-1]
                if f"{classname}_{fileid}" in kept_files:
                    filtered_files.append(file)
            print(f"Remaining files: {len(filtered_files)=}")
            all_files = filtered_files

        classes = None
        if params.class_cond:
            # Assume classes are the first part of the filename,
            # before an underscore. such as classcode_restoffilename.png
            print("ImageDataset: Compiling classes names")
            class_names = [os.path.normpath(path).split(os.sep)[-2] for path in all_files]
            sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
            classes = [sorted_classes[x] for x in class_names]
            print(f"ImageDataset: Found {len(sorted_classes):_} classes")

        self.resolution = params.image_size
        self.local_images = all_files
        self.local_classes = classes
        self.random_crop = params.random_crop
        self.random_flip = params.random_flip

    def __len__(self) -> int:
        return len(self.local_images)

    @jaxtyped(typechecker=typechecker)
    def __getitem__(
        self, idx: int
    ) -> tuple[Int[torch.Tensor, ""], Float[torch.Tensor, "3 h w"], dict]:
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.split == "train" or (self.split == "valid" and self.params.apply_augmentation_on_valid) or (self.split == "test" and self.params.apply_augmentation_on_test):
            if self.random_crop:
                arr = random_crop_arr(pil_image, self.resolution)
            else:
                arr = center_crop_arr(pil_image, self.resolution)

            if self.random_flip and random.random() < 0.5:
                arr = arr[:, ::-1]
        else:  # no augmentation, pil to numpy
            arr = center_crop_arr(pil_image, self.resolution)

        arr = arr.astype(np.float32) / 127.5 - 1
        arr = np.transpose(arr, [2, 0, 1])

        out_dict = {}
        if self.params.return_path:
            out_dict['img_path'] = path
        if self.local_classes is not None and self.params.return_class:
            out_dict["y"] = torch.tensor(self.local_classes[idx])

        t_idx = torch.tensor(idx).long()
        t_arr = torch.tensor(arr, dtype=torch.float32)

        return t_idx, t_arr, out_dict


class ImageDatasetDataModule(CustomDataModule):
    def _fetch_base_dataset(self) -> tuple[data.Dataset, data.Dataset, data.Dataset]:
        """
        Return train, valid and test dataset
        """
        params: ImageDatasetParams = self.p.data_params
        train_dataset = ImageDataset(params, split="train")
        valid_dataset = ImageDataset(params, split="valid")
        test_dataset = ImageDataset(params, split="test")

        print(
            "Data splitting parameters will be ignored for ImageDatasetDataset has the data are already splitted"
        )
        print("ImageDatasetDataset: train dataset size:", len(train_dataset))
        print("ImageDatasetDataset: val dataset size:", len(valid_dataset))
        print("ImageDatasetDataset: test dataset size:", len(test_dataset))

        return train_dataset, valid_dataset, test_dataset
