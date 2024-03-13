import torch
import torchvision.transforms as transforms


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
        return [q, k, torch.zeros(1)]  # torch.zeros : dummy
