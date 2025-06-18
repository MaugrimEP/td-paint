"""
File to load a dataset of pairs images/mask
"""
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
from tqdm import tqdm

from conf.dataset_params import (DatasetParams, ImageMaskDatasetParams,
                                 MaskParams)
from data.utils_ours import DatasetInpainting
from utils.utils import (display_mask, display_tensor, read_list_from_file,
                         write_list_to_file)


class ToOursDatasetWrapper(data.Dataset):
    def __init__(self, dataset: DatasetInpainting):
        self.dataset = dataset
        self.length = len(dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        items = self.dataset[index]
        classe, image, mask, save_name = items
        if classe != "":
            raise NotImplementedError()
        classe = 0


        # our dataset is supposed to return (img_idx, img, dict_info) + (mask, )

        dict_info = {
            #"y": classe,
            # "save_name": save_name,
        }
        mask = mask.int()

        return index, image, dict_info, mask


class ImageMaskDataModule(pl.LightningDataModule, ABC):
    def __init__(self, params: ImageMaskDatasetParams):
        super().__init__()
        self.test_dataset = None
        self.params = params.data_params
        self.params.worker = params.workers
        self.batch_size = params.batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        test_dataset = DatasetInpainting(
            images_path=self.params.init_image,
            masks_path=self.params.mask,
            class_cond=self.params.class_cond,
            outdir=self.params.outdir,
            blackiskeep=self.params.blackiskeep,
            split=self.params.split,
            max_split=self.params.max_split,
            noverwrite=self.params.noverwrite,
            img_extension=self.params.img_extension,
            image_size=self.params.image_size,
            prompt=self.params.prompt,
        )
        self.test_dataset = ToOursDatasetWrapper(test_dataset)

    def test_dataloader(self):
        dataset = self.test_dataset
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.params.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        return dataloader
