import os
from abc import ABC, abstractmethod
from typing import Optional, Union

import blobfile as bf
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms.functional as TF
from jaxtyping import Int
from PIL import Image
from torch.utils import data
from torch.utils.data import Subset, random_split
from torchvision import transforms

from conf.dataset_params import DatasetParams, MaskParams
from utils.utils import (display_mask, display_tensor, read_list_from_file,
                         write_list_to_file)


def filter_func(list_images: list[str], p: DatasetParams) -> list[int]:
    # return the list of indices that are already processed
    if p.filter_func == 'celebahq_cond':
        # eg. test_van_generate_cond_cond_4_img.png
        already_processed = []
        for image in list_images:
            if '_cond_' in image:
                already_processed.append(int( image.split('_')[-2]))
    elif p.filter_func == 'imagenet_cond':
        # eg. test_ema_generate_cond_cond_980_img_classe_332.png
        already_processed = []
        for image in list_images:
            if '_cond_' in image:
                already_processed.append(int( image.split('_')[-4]))
    else:
        raise ValueError(f"{p.filter_func=}")
                
    return already_processed


def list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(list_image_files_recursively(full_path))
    return results


class CustomDataModule(pl.LightningDataModule, ABC):
    def __init__(self, params: DatasetParams):
        super().__init__()
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.p = params
        self.batch_size = params.batch_size

    @abstractmethod
    def _fetch_base_dataset(self) -> tuple[data.Dataset, data.Dataset, data.Dataset]:
        """
        Return train, valid and test dataset
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        (
            base_train_dataset,
            base_valid_dataset,
            base_test_dataset,
        ) = self._fetch_base_dataset()

        train_dataset = MaskedDataset(base_train_dataset, self.p.mask_params)
        valid_dataset = MaskedDataset(base_valid_dataset, self.p.mask_params)
        test_dataset = MaskedDataset(base_test_dataset, self.p.mask_params)

        print(f"{len(train_dataset)=}")
        print(f"{len(valid_dataset)=}")
        print(f"{len(test_dataset)=}")

        if self.p.limit_train_prop is not None:
            print(f"[Limit train prop to {self.p.limit_train_prop=}, {self.p.limit_proportion_mode=}]")
            train_dataset = self.limite_size(
                dataset=train_dataset,
                limit=self.p.limit_train_prop,
                proportion_mode=self.p.limit_proportion_mode,
            )

        if self.p.limit_valid_prop is not None:
            print(f"[Limit valid prop to {self.p.limit_valid_prop=}, {self.p.limit_proportion_mode=}]")
            valid_dataset = self.limite_size(
                dataset=valid_dataset,
                limit=self.p.limit_valid_prop,
                proportion_mode=self.p.limit_proportion_mode,
            )

        if self.p.limit_test_prop is not None:
            print(f"[Limit test prop to {self.p.limit_test_prop=}, {self.p.limit_proportion_mode=}]")
            test_dataset = self.limite_size(
                dataset=test_dataset,
                limit=self.p.limit_test_prop,
                proportion_mode=self.p.limit_proportion_mode,
            )

        # we do it at the last bcs we set the mask, and that could change the mask index
        assert not (self.p.filter_from_results is not None and self.p.filter_from_file is not None), f"{self.p.filter_from_file=} {self.p.filter_from_results=} only one"

        already_done = []
        if self.p.filter_from_results is not None:
            print(f"Filter from results the test set: {self.p.filter_from_results=}")
            assert self.p.filter_func is not None, f"filter_func should not be None"
            assert os.path.isdir(self.p.filter_from_results), f"Trying to filter from a path that does not exist: {self.p.filter_from_results=}"
            imgs_test_paths = os.listdir(self.p.filter_from_results)
            already_done += filter_func(imgs_test_paths, self.p)
            print(f"Already done from results: {len(already_done)=}")
        
        if self.p.filter_from_file is not None:
            print(f"Filter from file the test set: {self.p.filter_from_file=}")
            assert os.path.isfile(self.p.filter_from_file), f"Trying to filter from a file that does not exist: {self.p.filter_from_file=}"
            already_done += read_list_from_file(self.p.filter_from_file)
            print(f"Already done from file: {len(already_done)=}")

        if self.p.inv_filter_from_file is not None:
            print(f"INV Filter from file the test set: {self.p.inv_filter_from_file=}")
            assert os.path.isfile(self.p.inv_filter_from_file), f"Trying to filter from a file that does not exist: {self.p.inv_filter_from_file=}"
            to_keep = read_list_from_file(self.p.inv_filter_from_file)
            to_remove = list(
                set(range(len(test_dataset))) - set(to_keep)
            )
            print(f"Number of sample to keep from file: {len(to_keep)=}")
            already_done += to_remove
            print(f"Already done from file: {len(already_done)=}")

        if len(already_done) > 0:
            print(f"Filtering from results the test set: {len(already_done)=} ... ")
            remaining_indexes = list(set(range(len(test_dataset))) - set(already_done))
            test_dataset = torch.utils.data.Subset(test_dataset, remaining_indexes)
            print(f"Dataset filtered new dataset length {len(test_dataset)=}")

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        
    def limite_size(
        self,
        dataset: data.Dataset,
        limit: Union[float, int, str],
        proportion_mode: str,
    ) -> data.Dataset:
        len_d = len(dataset)

        if proportion_mode == "frac":
            size = int(len_d * limit)
        elif proportion_mode == "perc":
            size = int(len_d * float(limit) / 100)
        elif proportion_mode == "abso":
            size = int(limit)
        else:
            raise ValueError(f"{self.p.proportion_mode=}")
        
        assert size <= len_d, f"{size=} > {len_d=}"
        
        subdataset = torch.utils.data.Subset(dataset, list(range(size)))
        return subdataset

    def train_dataloader(self):
        dataset = self.train_dataset
        if (
            self.p.use_min_for_batch_size
            and self.p.drop_last_train
            and self.batch_size > len(dataset)
        ):
            print(
                f"[DropLast + Train dataset size = {len(dataset)} < {self.batch_size=}] => set batch size to dataset size"
                f"this ensure that we do not have an empty dataset with drop last = True"
            )
            self.batch_size = len(dataset)

        return data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.p.workers,
            pin_memory=self.p.pin_memory,
            drop_last=self.p.drop_last_train,
        )

    def val_dataloader(self):
        dataset = self.valid_dataset
        
        return data.DataLoader(
            dataset,
            batch_size=self.p.batch_size_val,
            shuffle=self.p.shuffle_val,
            num_workers=self.p.workers,
            pin_memory=self.p.pin_memory,
            drop_last=self.p.drop_last_valid,
        )

    def test_dataloader(self):
        dataset = self.test_dataset
        
        return data.DataLoader(
            dataset,
            batch_size=self.p.batch_size_test,
            shuffle=self.p.shuffle_test,
            num_workers=self.p.workers,
            pin_memory=self.p.pin_memory,
            drop_last=self.p.drop_last_test,
        )

    def split_dataset(self, dataset: data.Dataset):
        """
        Instantiate the datasets and split them into train, val, test
        """
        len_d = len(dataset)
        proportion_mode = self.p.proportion_mode

        if proportion_mode == "frac":
            train_size = int(len_d * self.p.train_prop)
            valid_size = int(len_d * self.p.valid_prop)
            test_size = len_d - train_size - valid_size
        elif proportion_mode == "perc":
            train_size = int(len_d * self.p.train_prop / 100)
            valid_size = int(len_d * self.p.valid_prop / 100)
            test_size = len_d - train_size - valid_size
        elif proportion_mode == "abso":
            train_size = self.p.train_prop
            valid_size = self.p.valid_prop
            test_size = self.p.test_prop
        else:
            raise ValueError(f"{self.p.proportion_mode=}")

        self.train_size = train_size
        self.valid_size = valid_size
        self.test_size = test_size
        print(
            f"""
            Splitting size: {train_size=} {valid_size=} {test_size=}
        """
        )

        # split the dataset
        if self.p.file_path is None or not os.path.exists(self.p.file_path):
            train_dataset, valid_dataset, test_dataset = random_split(
                dataset, [train_size, valid_size, test_size]
            )

            if self.p.file_path is not None and not os.path.exists(self.p.file_path):
                ids_list = (
                    list(train_dataset.indices)
                    + list(valid_dataset.indices)
                    + list(test_dataset.indices)
                )
                print(
                    f"[CustomDataModule:split_dataset] Write indices to {self.p.file_path}"
                )
                write_list_to_file(file_path=self.p.file_path, integer_list=ids_list)
        else:
            indices = read_list_from_file(self.p.file_path)
            assert len(indices) == len(dataset), f"{len(indices)=} != {len(dataset)=}"
            indices_train = indices[:train_size]
            indices_valid = indices[train_size : train_size + valid_size]
            indices_test = indices[
                train_size + valid_size : train_size + valid_size + test_size
            ]

            train_dataset = Subset(dataset, indices_train)
            valid_dataset = Subset(dataset, indices_valid)
            test_dataset = Subset(dataset, indices_test)

        print("split_dataset >")
        print(f"{len(train_dataset)=}")
        print(f"{len(valid_dataset)=}")
        print(f"{len(test_dataset)=}")

        return train_dataset, valid_dataset, test_dataset


class MaskedDataset(data.Dataset):
    def __init__(
        self,
        dataset: data.Dataset,
        mask_params: MaskParams,
    ):
        super().__init__()
        self.dataset = dataset
        self.mask_params = mask_params

        mask_dir = os.path.expanduser(mask_params.mask_root)
        # get the list of folder in mask_dir
        self.mask_folders = [
            os.path.join(mask_dir, mask_type) for mask_type in mask_params.mask_type
        ]
        all_masks = []
        for mask_folder in self.mask_folders:
            all_masks.extend(list_image_files_recursively(mask_folder))

        self.all_mask = sorted(all_masks)

        self.resize_mask = transforms.Resize([mask_params.height, mask_params.width], TF.InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.dataset)

    def imread(self, path):
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")
        return pil_image

    def get_mask(self, idx: int) -> tuple[Int[torch.Tensor, "b 1 h w"], str]:
        mask_indice = idx % len(self.all_mask)
        mask_path = self.all_mask[mask_indice]
        pil_mask = self.imread(mask_path)
        mask = np.array(pil_mask)
        mask = mask[:, :, :1]
        mask = mask / 255.0
        mask = mask.transpose(2, 0, 1)
        mask = torch.from_numpy(mask).long()
        return mask, mask_path

    def __getitem__(self, idx: int):
        img_idx, img, dict_info = self.dataset[idx]
        mask, mask_path = self.get_mask(idx)
        if self.mask_params.return_path:
            dict_info["mask_path"] = mask_path

        # resize the mask to the data dimension
        mask = self.resize_mask(mask)  # mask are in [0,1]

        return (img_idx, img, dict_info) + (mask, )
