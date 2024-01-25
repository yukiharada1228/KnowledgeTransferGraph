# Import packages
import argparse
import os

import optuna
import torch
import torch.nn as nn
from ktg import KnowledgeTransferGraph, Node, gates
from ktg.losses import KLDivLossSoftTarget, TotalLoss
from ktg.models import cifar_models
from ktg.utils import (AverageMeter, WorkerInitializer, load_checkpoint,
                       set_seed)
from optuna.storages import JournalFileStorage, JournalStorage
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42)
parser.add_argument("--num-nodes", default=2)
parser.add_argument(
    "--models",
    default=["resnet20", "resnet32", "resnet56", "resnet110"],
    nargs="*",
    type=str,
)

args = parser.parse_args()
manualSeed = args.seed
num_nodes = args.num_nodes
models_name = args.models


def objective(trial):
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
    T = 1
    nodes = []
    for i in range(num_nodes):
        gates_list = []
        criterions = []
        for j in range(num_nodes):
            if i == j:
                criterions.append(nn.CrossEntropyLoss().cuda())
            else:
                criterions.append(KLDivLossSoftTarget(T=T).cuda())
            gate_name = trial.suggest_categorical(
                f"{i}_{j}_gate",
                [
                    "ThroughGate",
                    "CutoffGate",
                    "PositiveLinearGate",
                    "NegativeLinearGate",
                ],
            )
            gate = getattr(gates, gate_name)(max_epoch)
            gates_list.append(gate)
        if i == 0:
            model_name = models_name[0]
        else:
            model_name = trial.suggest_categorical(f"{i}_model", models_name)
        model = getattr(cifar_models, model_name)(num_classes).cuda()
        if all(gate.__class__.__name__ == "CutoffGate" for gate in gates_list):
            load_checkpoint(
                model=model, save_dir=f"checkpoint/pre-train/{model_name}", is_best=True
            )
        writer = SummaryWriter(f"runs/dcl/{trial.number:04}/{i}_{model_name}")
        save_dir = f"checkpoint/dcl/{trial.number:04}/{i}_{model_name}"
        optimizer = getattr(torch.optim, optim_setting["name"])(
            model.parameters(), **optim_setting["args"]
        )
        scheduler = getattr(torch.optim.lr_scheduler, scheduler_setting["name"])(
            optimizer, **scheduler_setting["args"]
        )
        criterion = TotalLoss(criterions, gates=gates_list)

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
        trial=trial,
    )
    graph.train()


if __name__ == "__main__":
    # Cteate study object
    optuna_dir = "optuna/dcl"
    os.makedirs(optuna_dir, exist_ok=True)
    storage = JournalStorage(JournalFileStorage(os.path.join(optuna_dir, "optuna.log")))
    study = optuna.create_study(
        storage=storage,
        study_name="experiment",
        sampler=optuna.samplers.TPESampler(multivariate=True),
        pruner=optuna.pruners.SuccessiveHalvingPruner(),
        direction="maximize",
        load_if_exists=True,
    )
    # Start optimization
    study.optimize(objective, n_trials=1500)
