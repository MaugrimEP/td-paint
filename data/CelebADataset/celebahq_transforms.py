from torchvision.transforms import transforms

from conf.dataset_params import CelebAParams


def get_image_transform(params: CelebAParams):
    img_transform = [
        transforms.Resize([params.image_size, params.image_size]),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
    ]

    return transforms.Compose(img_transform)


def get_image_augmentation(params: CelebAParams):
    img_transform = []
    if params.random_flip:
        img_transform += [
            transforms.RandomHorizontalFlip(),
        ]

    return transforms.Compose(img_transform)
