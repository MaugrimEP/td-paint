from conf.dataset_params import DatasetParams
from data.CelebADataset.CelebADataset import CelebADataModule
from data.ImageDatasetInpaint.ImageDataset import ImageDatasetDataModule
from data.ImageMaskDataset import ImageMaskDataModule
from data.ImageNetDataset.ImageNetDataset import ImageNetDataModule


def get_dm(params: DatasetParams):
    dataset_name = params.data_params.name
    if dataset_name in ["celeba"]:
        return CelebADataModule(params)
    elif dataset_name in ["imagenet"]:
        return ImageNetDataModule(params)
    elif dataset_name in ["imagedataset"]:
        return ImageDatasetDataModule(params)
    elif dataset_name in ["imagemask"]:
        return ImageMaskDataModule(params)
    else:
        raise Exception(f"Dataset type not available: {dataset_name=}")
