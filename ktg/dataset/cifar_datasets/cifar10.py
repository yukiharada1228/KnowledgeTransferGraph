# Import packages
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from ..custom_dataset import CustomDataset


def get_datasets():
    dataset = datasets.CIFAR10(root="data", train=True, download=True)
    lengths = [40000, 10000]
    subsets = random_split(dataset, lengths)
    loader = DataLoader(
        CustomDataset(subsets[0], transform=transforms.ToTensor()), batch_size=64
    )

    h, w = 0, 0
    for batch_idx, (inputs, _) in enumerate(loader):
        inputs = inputs.cuda()
        if batch_idx == 0:
            h, w = inputs.size(2), inputs.size(3)
            chsum = inputs.sum(dim=(0, 2, 3), keepdim=True)
        else:
            chsum += inputs.sum(dim=(0, 2, 3), keepdim=True)
    mean = chsum / len(subsets[0]) / h / w
    chsum = None
    for batch_idx, (inputs, _) in enumerate(loader):
        inputs = inputs.cuda()
        if batch_idx == 0:
            chsum = (inputs - mean).pow(2).sum(dim=(0, 2, 3), keepdim=True)
        else:
            chsum += (inputs - mean).pow(2).sum(dim=(0, 2, 3), keepdim=True)
    std = torch.sqrt(chsum / (len(subsets[0]) * h * w - 1))

    mean = mean.view(-1).cpu().numpy()
    std = std.view(-1).cpu().numpy()
    print("mean: %s" % mean)
    print("std: %s" % std)

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_dataset = CustomDataset(subsets[0], transform=train_transform)
    val_dataset = CustomDataset(subsets[1], transform=test_transform)
    test_dataset = datasets.CIFAR10(
        root="data", train=True, download=True, transform=test_transform
    )
    return train_dataset, val_dataset, test_dataset
