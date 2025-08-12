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


class SimCLRTransforms(object):
    # Appendix A (SimCLR) 準拠の拡張をそのまま適用
    # - RandomResizedCrop: scale=(0.08, 1.0), ratio=(3/4, 4/3)
    # - RandomHorizontalFlip: p=0.5
    # - ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s) を p=0.8
    # - RandomGrayscale: p=0.2
    # - GaussianBlur: p=0.5, sigma=[0.1, 2.0], kernel_size=高さ/幅の10%（奇数, 最小3）
    # - ToTensor
    def __init__(self, input_size=32, s: float = 0.5, include_blur: bool = False):
        kernel_size = max(3, int(round(0.1 * input_size)))
        if kernel_size % 2 == 0:
            kernel_size += 1

        color_jitter = transforms.ColorJitter(
            brightness=0.8 * s, contrast=0.8 * s, saturation=0.8 * s, hue=0.2 * s
        )

        simclr_ops = [
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ]
        if include_blur:
            simclr_ops.append(
                transforms.RandomApply(
                    [
                        transforms.GaussianBlur(
                            kernel_size=kernel_size, sigma=(0.1, 2.0)
                        )
                    ],
                    p=0.5,
                )
            )
        simclr_ops.append(transforms.ToTensor())

        self.train_transform = transforms.Compose(simclr_ops)

    def __call__(self, x):
        q = self.train_transform(x)
        k = self.train_transform(x)
        return [q, k]


class MoCoTransforms(object):
    # https://github.com/facebookresearch/moco/blob/main/main_moco.py#L327
    def __init__(self, input_size=32):
        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0)),
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


class BYOLTransforms(object):
    # https://github.com/deepmind/deepmind-research/blob/master/byol/utils/augmentations.py
    def __init__(self, input_size=32):
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


class DINOTransforms(object):
    # https://github.com/facebookresearch/dino/blob/cb711401860da580817918b9167ed73e3eef3dcf/main_dino.py#L419
    def __init__(
        self,
        size_crops=[32, 16],
        local_crops_number=6,
        global_crops_scale=(0.14, 1.0),
        local_crops_scale=(0.05, 0.14),
    ):
        self.train_transform_global1 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size_crops[0], scale=global_crops_scale, interpolation=Image.BICUBIC
                ),
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
                transforms.ToTensor(),
            ]
        )
        self.train_transform_global2 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size_crops[0], scale=global_crops_scale, interpolation=Image.BICUBIC
                ),
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
        self.train_transform_local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size_crops[1], scale=local_crops_scale, interpolation=Image.BICUBIC
                ),
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
                transforms.ToTensor(),
            ]
        )
        self.local_crops_number = local_crops_number

    def __call__(self, x):
        # 1つの画像に対して複数のデータ増幅を適用した画像を取得
        multi_crops = []
        q = self.train_transform_global1(x)
        k = self.train_transform_global2(x)
        multi_crops = []
        for _ in range(self.local_crops_number):
            multi_crops.append(self.train_transform_local(x))
        return [q, k, multi_crops]
