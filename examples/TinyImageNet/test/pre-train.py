# Import packages
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from ktg import Edges, KnowledgeTransferGraph, Node
from ktg.dataset.tinyimagenet import TinyImageNet
from ktg.gates import ThroughGate
from ktg.models import cifar_models
from ktg.utils import AverageMeter, WorkerInitializer, set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42)
parser.add_argument("--lr", default=0.1)
parser.add_argument("--wd", default=5e-4)
parser.add_argument("--model", default="resnet18")

args = parser.parse_args()
manualSeed = int(args.seed)
model_name = args.model
lr = float(args.lr)
wd = float(args.wd)

# Fix the seed value
set_seed(manualSeed)

# Prepare the CIFAR-100 for training
batch_size = 64
num_workers = 10


def get_datasets():
    dataset = TinyImageNet("tiny-imagenet-200", split="train", transform=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=64)
    h, w = 0, 0
    for batch_idx, (inputs, _) in enumerate(loader):
        inputs = inputs.cuda()
        if batch_idx == 0:
            h, w = inputs.size(2), inputs.size(3)
            chsum = inputs.sum(dim=(0, 2, 3), keepdim=True)
        else:
            chsum += inputs.sum(dim=(0, 2, 3), keepdim=True)
    mean = chsum / len(dataset) / h / w
    chsum = None
    for batch_idx, (inputs, _) in enumerate(loader):
        inputs = inputs.cuda()
        if batch_idx == 0:
            chsum = (inputs - mean).pow(2).sum(dim=(0, 2, 3), keepdim=True)
        else:
            chsum += (inputs - mean).pow(2).sum(dim=(0, 2, 3), keepdim=True)
    std = torch.sqrt(chsum / (len(dataset) * h * w - 1))

    mean = mean.view(-1).cpu().numpy()
    std = std.view(-1).cpu().numpy()
    print("mean: %s" % mean)
    print("std: %s" % std)

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(64, padding=4),
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

    train_dataset = TinyImageNet("tiny-imagenet-200", split="train", transform=train_transform)
    test_dataset = TinyImageNet("tiny-imagenet-200", split="val", transform=test_transform)
    return train_dataset, test_dataset


train_dataset, val_dataset = get_datasets()

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True,
    worker_init_fn=WorkerInitializer(manualSeed).worker_init_fn,
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=False,
    worker_init_fn=WorkerInitializer(manualSeed).worker_init_fn,
)

# Prepare for training
max_epoch = 200

optim_setting = {
    "name": "SGD",
    "args": {
        "lr": lr,
        "momentum": 0.9,
        "weight_decay": wd,
        "nesterov": True,
    },
}
scheduler_setting = {
    "name": "CosineAnnealingLR",
    "args": {"T_max": max_epoch, "eta_min": 0.0},
}

num_classes = 200
nodes = []

gates = [ThroughGate(max_epoch)]
criterions = [nn.CrossEntropyLoss()]
model = getattr(cifar_models, model_name)(num_classes).cuda()
writer = SummaryWriter(f"runs/pre-train/{model_name}")
save_dir = f"checkpoint/pre-train/{model_name}"
optimizer = getattr(torch.optim, optim_setting["name"])(
    model.parameters(), **optim_setting["args"]
)
scheduler = getattr(torch.optim.lr_scheduler, scheduler_setting["name"])(
    optimizer, **scheduler_setting["args"]
)
edges = Edges(criterions, gates)

node = Node(
    model=model,
    writer=writer,
    scaler=torch.cuda.amp.GradScaler(),
    save_dir=save_dir,
    optimizer=optimizer,
    scheduler=scheduler,
    edges=edges,
    loss_meter=AverageMeter(),
    top1_meter=AverageMeter(),
)
nodes.append(node)

graph = KnowledgeTransferGraph(
    nodes=nodes,
    max_epoch=max_epoch,
    train_dataloader=train_dataloader,
    test_dataloader=val_dataloader,
)
graph.train()
