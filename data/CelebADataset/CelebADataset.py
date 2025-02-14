import os
import os.path as osp

import torch
import torch.utils.data as data
from jaxtyping import Float, Int
from PIL import Image

from conf.dataset_params import CelebAParams
from data.CelebADataset.celebahq_transforms import get_image_transform, get_image_augmentation
from data.UtilsDataset import CustomDataModule
from utils.utils import display_mask, display_tensor

SRC_FACE = "CelebA-HQ-img/"


def fromFilenameGetNumber(filename: str) -> int:
    return int(filename.split(".")[0])


class CelebAHQDataset(data.Dataset):
    def __init__(
        self,
        params: CelebAParams,
        split: str,
    ):
        super(CelebAHQDataset, self).__init__()
        assert split in ['train', 'valid', 'test'], f'Split should be train, valid or test, got {split=}'
        self.params = params
        self.split = split

        imgs = sorted(
            os.listdir(os.path.join(params.root, SRC_FACE)), key=fromFilenameGetNumber
        )
        file = {
            'train': params.train_file,
            'valid': params.valid_file,
            'test': params.test_file,
        }[split]
        with open(file, 'r') as f:
            kept_idx = [int(i) for i in f.readline().split(',')]  # only one line, list of int
        imgs = [imgs[i] for i in kept_idx]
        self.imgs = imgs

        self.image_transform = get_image_transform(params)
        self.image_augmentation = None
        if (split == 'train') or (split == 'valid' and params.apply_augmentation_on_valid) or (split == 'test' and params.apply_augmentation_on_test):
            self.image_augmentation = get_image_augmentation(params)
            

    def __getitem__(
        self, idx: int
    ) -> tuple[Int[torch.Tensor, "1"], Float[torch.Tensor, "3 h w"], dict]:
        img_path = self.imgs[idx]
        img_number = fromFilenameGetNumber(img_path)
        img = Image.open(osp.join(self.params.root, SRC_FACE, img_path))
        img = self.image_transform(img) if self.image_transform is not None else img
        img = self.image_augmentation(img) if self.image_augmentation is not None else img

        # img and sketch is in [0, 1], put them in [-1, 1]
        img = img * 2 - 1

        return torch.tensor(img_number), img, dict()

    def __len__(self) -> int:
        return len(self.imgs)


class CelebADataModule(CustomDataModule):
    def _fetch_base_dataset(self) -> tuple[data.Dataset, data.Dataset, data.Dataset]:
        """
        Return train, valid and test dataset
        """
        params: CelebAParams = self.p.data_params

        train_dataset = CelebAHQDataset(params, split='train')
        valid_dataset = CelebAHQDataset(params, split='valid')
        test_dataset = CelebAHQDataset(params, split='test')

        return train_dataset, valid_dataset, test_dataset
