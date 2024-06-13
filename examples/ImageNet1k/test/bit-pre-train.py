# Import packages
import argparse

import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from ktg import Edges, KnowledgeTransferGraph, Node
from ktg.gates import ThroughGate
from ktg.models import imagenet_models
from ktg.utils import AverageMeter, WorkerInitializer, set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42)
parser.add_argument("--lr", default=0.45)
parser.add_argument("--wd", default=3e-5)
parser.add_argument("--model", default="bit_resnet152_b158")

args = parser.parse_args()
manualSeed = int(args.seed)
model_name = args.model
lr = float(args.lr)
wd = float(args.wd)

# Fix the seed value
set_seed(manualSeed)

# Prepare the ImageNet1k for training
batch_size = 256
num_workers = 10

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_dataloader = DataLoader(
    datasets.ImageFolder(
        "./dataset/imagenet_data/train",
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    ),
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    worker_init_fn=WorkerInitializer(manualSeed).worker_init_fn,
)

val_dataloader = DataLoader(
    datasets.ImageFolder(
        "./dataset/imagenet_data/val",
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    ),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    worker_init_fn=WorkerInitializer(manualSeed).worker_init_fn,
)

# Prepare for training
max_epoch = 90

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

num_classes = 1000
nodes = []

gates = [ThroughGate(max_epoch)]
criterions = [nn.CrossEntropyLoss()]
model = torch.nn.DataParallel(getattr(imagenet_models, model_name)(num_classes)).cuda()
writer = SummaryWriter(f"runs/pre-train/{model_name}")
save_dir = f"checkpoint/pre-train/{model_name}"
optimizer = getattr(torch.optim, optim_setting["name"])(
    model.parameters(), **optim_setting["args"]
)
scheduler = getattr(torch.optim.lr_scheduler, scheduler_setting["name"])(
    optimizer, **scheduler_setting["args"]
)
edges = Edges(criterions, gates)

class WeightDecayScheduler:
    def __init__(self, optimizer, total_steps, decay_start_step, last_epoch=-1):
        self.optimizer = optimizer
        self.decay_start_step = decay_start_step
        self.total_steps = total_steps
        self.last_epoch = last_epoch
        self.base_weight_decays = [
            group["weight_decay"] for group in optimizer.param_groups
        ]

    def step(self, current_step):
        if current_step >= self.decay_start_step:
            weight_decay = 0.0
        else:
            weight_decay = self.base_weight_decays[
                0
            ]  # Assuming all groups have the same weight decay
        for param_group, base_weight_decay in zip(
            self.optimizer.param_groups, self.base_weight_decays
        ):
            param_group["weight_decay"] = weight_decay

start_rate = 2
wdscheduler = WeightDecayScheduler(
    optimizer=optimizer,
    total_steps=max_epoch,
    decay_start_step=max_epoch // start_rate,
)

node = Node(
    model=model,
    writer=writer,
    scaler=torch.cuda.amp.GradScaler(),
    save_dir=save_dir,
    optimizer=optimizer,
    scheduler=scheduler,
    wdscheduler=wdscheduler,
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
