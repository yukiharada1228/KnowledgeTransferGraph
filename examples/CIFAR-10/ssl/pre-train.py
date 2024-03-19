# Import packages
import argparse
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from ktg import Edges, KnowledgeTransferGraph, Node, losses
from ktg.dataset.cifar_datasets.cifar10 import get_datasets
from ktg.gates import ThroughGate
from ktg.models import cifar_models, projector, ssl_models
from ktg.transforms import ssl_transforms
from ktg.utils import (LARS, AverageMeter, KNNValidation, WorkerInitializer,
                       get_cosine_schedule_with_warmup, set_seed)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42)
parser.add_argument("--model", default="resnet18")
parser.add_argument("--ssl", default="SimCLR")
parser.add_argument("--transforms", default="DINO")
parser.add_argument("--projector", default="BarlowTwins")

args = parser.parse_args()
manual_seed = args.seed
model_name = args.model
ssl_name = args.ssl
transforms_name = args.transforms
projector_name = args.projector

# Fix the seed value
set_seed(manual_seed)

# Prepare the CIFAR-10 for training
batch_size = 512
num_workers = 10

train_dataset, val_dataset, _ = get_datasets()
transform = getattr(ssl_transforms, f"{transforms_name}Transforms")()
train_dataset.transform = transform
val_dataset.transform = transform

knn_train_dataset = deepcopy(train_dataset)
knn_val_dataset = deepcopy(val_dataset)
knn_train_dataset.transform = transforms.ToTensor()
knn_val_dataset.transform = transforms.ToTensor()

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True,
    worker_init_fn=WorkerInitializer(manual_seed).worker_init_fn,
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=False,
    worker_init_fn=WorkerInitializer(manual_seed).worker_init_fn,
)

# Prepare for training
max_epoch = 800

optim_setting = {
    "name": "LARS",
    "args": {
        "lr": 0.3 * (batch_size / 256),
        "weight_decay": 10e-6,
        "momentum": 0.9,
        "eta": 0.001,
        "weight_decay_filter": False,
        "lars_adaptation_filter": False,
    },
}
scheduler_setting = {
    "name": "get_cosine_schedule_with_warmup",
    "args": {
        "num_warmup_steps": 10,
        "num_training_steps": max_epoch,
        "num_cycles": 0.5,
        "last_epoch": -1,
    },
}

nodes = []
gates = [ThroughGate(max_epoch)]
model = getattr(ssl_models, ssl_name)(
    encoder_func=getattr(cifar_models, model_name),
    batch_size=batch_size,
    projector_func=getattr(projector, f"{projector_name}Projector"),
).cuda()
criterions = [getattr(losses, "SSLLoss")()]
writer = SummaryWriter(
    f"runs/pre-train/{model_name}/{projector_name}/{transforms_name}/{ssl_name}"
)
save_dir = (
    f"checkpoint/pre-train/{model_name}/{projector_name}/{transforms_name}/{ssl_name}"
)
optimizer = LARS(model.parameters(), **optim_setting["args"])
scheduler = get_cosine_schedule_with_warmup(optimizer, **scheduler_setting["args"])
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
    eval=KNNValidation(
        model,
        knn_train_dataset,
        knn_val_dataset,
        K=20,
    ),
)
nodes.append(node)

graph = KnowledgeTransferGraph(
    nodes=nodes,
    max_epoch=max_epoch,
    train_dataloader=train_dataloader,
    test_dataloader=val_dataloader,
)
graph.train()
