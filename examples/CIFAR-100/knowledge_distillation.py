# Import packages
import argparse

import torch
import torch.nn as nn
from ktg import KnowledgeTransferGraph, Node
from ktg.gates import CutoffGate, ThroughGate
from ktg.losses import CrossEntropyLossSoftTarget, TotalLoss
from ktg.models import cifar_models
from ktg.utils import (AverageMeter, WorkerInitializer, load_checkpoint,
                       set_seed)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42)
parser.add_argument("--models", default=["resnet20", "resnet32"], nargs="*", type=str)

args = parser.parse_args()
manualSeed = args.seed
models_name = args.models

# Fix the seed value
set_seed(manualSeed)

# Prepare the CIFAR-100 for training
batch_size = 128
num_workers = 4


cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)

train_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std),
    ]
)
test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std),
    ]
)

train_dataset = datasets.CIFAR100(
    root="data", train=True, download=True, transform=train_transform
)
test_dataset = datasets.CIFAR100(
    root="data", train=False, download=True, transform=test_transform
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True,
    worker_init_fn=WorkerInitializer(manualSeed).worker_init_fn,
)
test_dataloader = DataLoader(
    test_dataset,
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
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "nesterov": True,
    },
}
scheduler_setting = {
    "name": "CosineAnnealingLR",
    "args": {"T_max": max_epoch, "eta_min": 0.0},
}

num_classes = 100
T = 10
nodes = []
for i, model_name in enumerate(models_name):
    gates = []
    criterions = []
    for j in range(len(models_name)):
        if i == 0:
            gates.append(ThroughGate(max_epoch))
        else:
            gates.append(CutoffGate(max_epoch))
        if i == j:
            criterions.append(nn.CrossEntropyLoss().cuda())
        else:
            criterions.append(CrossEntropyLossSoftTarget(T=T).cuda())
    model = getattr(cifar_models, model_name)(num_classes).cuda()
    if i == 1:
        # If i is a teacher model
        load_checkpoint(
            model=model, save_dir=f"checkpoint/pre-train/{model_name}", is_best=True
        )
    writer = SummaryWriter(
        f"runs/kd/{models_name[0]}_{models_name[1]}/{i}_{model_name}"
    )
    save_dir = f"checkpoint/kd/{models_name[0]}_{models_name[1]}/{i}_{model_name}"
    optimizer = getattr(torch.optim, optim_setting["name"])(
        model.parameters(), **optim_setting["args"]
    )
    scheduler = getattr(torch.optim.lr_scheduler, scheduler_setting["name"])(
        optimizer, **scheduler_setting["args"]
    )
    criterion = TotalLoss(criterions, gates)

    node = Node(
        model=model,
        writer=writer,
        scaler=torch.cuda.amp.GradScaler(),
        save_dir=save_dir,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        loss_meter=AverageMeter(),
        top1_meter=AverageMeter(),
    )
    nodes.append(node)

graph = KnowledgeTransferGraph(
    nodes=nodes,
    max_epoch=max_epoch,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
)
graph.train()
