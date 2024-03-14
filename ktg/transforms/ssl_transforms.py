import random

import torchvision.transforms as transforms
from PIL import Image, ImageOps


class Solarization(object):
    # https://github.com/facebookresearch/barlowtwins/blob/main/main.py#L279
    # https://github.com/facebookresearch/dino/blob/cb711401860da580817918b9167ed73e3eef3dcf/utils.py#L57
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class SimSiamTransforms(object):
    # https://github.com/facebookresearch/simsiam/blob/a7bc1772896d0dad0806c51f0bb6f3b16d290468/main_simsiam.py#L228
    def __init__(self, input_size=32):
        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    input_size, scale=(0.2, 1.0)
                ),  # Default: interpolation=InterpolationMode.BILINEAR
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )

    def __call__(self, x):
        # 1つの画像に対して２つのデータ増幅を適用した画像を取得
        q = self.train_transform(x)
        k = self.train_transform(x)
        return [q, k]


class SwAVTransforms(object):
    # https://github.com/facebookresearch/swav/blob/main/src/multicropdataset.py
    def __init__(
        self,
        size_crops=[32, 16],
        nmb_crops=[2, 6],
        min_scale_crops=[0.14, 0.05],
        max_scale_crops=[1.0, 0.14],
    ):
        trans = []
        for i in range(len(size_crops)):
            trans.extend(
                [
                    transforms.Compose(
                        [
                            transforms.RandomResizedCrop(
                                size_crops[i],
                                scale=(min_scale_crops[i], max_scale_crops[i]),
                            ),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.RandomApply(
                                [transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8
                            ),
                            transforms.RandomGrayscale(p=0.2),
                            transforms.ToTensor(),
                        ]
                    )
                ]
                * nmb_crops[i]
            )
        self.train_transform = trans

    def __call__(self, x):
        # 1つの画像に対して複数のデータ増幅を適用した画像を取得
        multi_crops = list(map(lambda trans: trans(x), self.train_transform))
        return [multi_crops[0], multi_crops[1], multi_crops[2:]]


class BarlowTwinsTransforms(object):
    # https://github.com/facebookresearch/barlowtwins/blob/main/main.py#L292
    def __init__(
        self,
        input_size=32,
    ):
        self.train_transform_a = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    input_size, interpolation=Image.BICUBIC
                ),  # Default: scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                Solarization(p=0.0),
                transforms.ToTensor(),
            ]
        )
        self.train_transform_b = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    input_size, interpolation=Image.BICUBIC
                ),  # Default: scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                Solarization(p=0.2),
                transforms.ToTensor(),
            ]
        )

    def __call__(self, x):
        # 1つの画像に対して２つのデータ増幅を適用した画像を取得
        q = self.train_transform_a(x)
        k = self.train_transform_b(x)
        return [q, k]
