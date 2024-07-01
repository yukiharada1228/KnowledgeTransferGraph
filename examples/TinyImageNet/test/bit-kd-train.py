# Import packages
import argparse
import os

import optuna
import torch
import torch.nn as nn
from ktg.dataset.tinyimagenet import TinyImageNet
from optuna.storages import JournalFileStorage, JournalStorage
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from ktg import Edges, KnowledgeTransferGraph, Node, gates
from ktg.losses import KLDivLoss
from ktg.models import cifar_models
from ktg.utils import (AverageMeter, WorkerInitializer,
                       get_cosine_schedule_with_warmup, load_checkpoint,
                       set_seed)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42)
parser.add_argument("--num-nodes", default=2)
parser.add_argument("--n_trials", default=10)
parser.add_argument(
    "--models",
    default=["bit_resnet50_b158", "resnet50"],
    nargs="*",
    type=str,
)
parser.add_argument(
    "--gates",
    default=["ThroughGate", "CutoffGate"],
    nargs="*",
    type=str,
)

args = parser.parse_args()
manualSeed = int(args.seed)
num_nodes = int(args.num_nodes)
n_trials = int(args.n_trials)
models_name = args.models
gates_name = args.gates


def objective(trial):
    # Fix the seed value
    set_seed(manualSeed)

    # Prepare the TinyImageNet for training
    batch_size = 64
    num_workers = 10

    def get_datasets():
        dataset = TinyImageNet(
            "tiny-imagenet-200", split="train", transform=transforms.ToTensor()
        )
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

        train_dataset = TinyImageNet(
            "tiny-imagenet-200", split="train", transform=train_transform
        )
        test_dataset = TinyImageNet(
            "tiny-imagenet-200", split="val", transform=test_transform
        )
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

    num_classes = 200
    nodes = []
    for i in range(num_nodes):
        gates_list = []
        criterions = []
        for j in range(num_nodes):
            if i == j:
                criterions.append(nn.CrossEntropyLoss())
            else:
                criterions.append(KLDivLoss())
            if i == 1:
                gate_name = trial.suggest_categorical(
                    f"{i}_{j}_gate",
                    gates_name[1:],
                )
            else:
                gate_name = trial.suggest_categorical(
                    f"{i}_{j}_gate",
                    gates_name[0:1],
                )
            gate = getattr(gates, gate_name)(max_epoch)
            gates_list.append(gate)
        if i == 0:
            model_name = trial.suggest_categorical(f"{i}_model", models_name[0:1])
        else:
            model_name = trial.suggest_categorical(f"{i}_model", models_name[1:])
        model = getattr(cifar_models, model_name)(num_classes).cuda()
        if (
            all(gate.__class__.__name__ == "CutoffGate" for gate in gates_list)
            and i != 0
        ):
            load_checkpoint(
                model=model, save_dir=f"checkpoint/pre-train/{model_name}", is_best=True
            )
        writer = SummaryWriter(
            f"runs/bit_kd_{num_nodes}/{trial.number:04}/{i}_{model_name}"
        )
        save_dir = f"checkpoint/bit_kd_{num_nodes}/{trial.number:04}/{i}_{model_name}"

        if "bit" in model_name:
            optim_setting = {
                "name": "SGD",
                "args": {
                    "lr": 0.45422698806566725 / 2,
                    "momentum": 0.9,
                    "weight_decay": 7.310284744110568e-05,
                    "nesterov": True,
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
            decay_start_step = trial.suggest_int('decay_start_step', 0, max_epoch)

            class WeightDecayScheduler:
                def __init__(
                    self, optimizer, total_steps, decay_start_step, last_epoch=-1
                ):
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

            optimizer = getattr(torch.optim, optim_setting["name"])(
                model.parameters(), **optim_setting["args"]
            )
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, **scheduler_setting["args"]
            )
            wdscheduler = WeightDecayScheduler(
                optimizer=optimizer,
                total_steps=max_epoch,
                decay_start_step=decay_start_step,
            )
        else:
            optim_setting = {
                "name": "SGD",
                "args": {
                    "lr": 0.1,
                    "momentum": 0.9,
                    "weight_decay": 1e-4,
                    "nesterov": True,
                },
            }
            scheduler_setting = {
                "name": "CosineAnnealingLR",
                "args": {"T_max": max_epoch, "eta_min": 0.0},
            }
            wdscheduler = None
            optimizer = getattr(torch.optim, optim_setting["name"])(
                model.parameters(), **optim_setting["args"]
            )
            scheduler = getattr(torch.optim.lr_scheduler, scheduler_setting["name"])(
                optimizer, **scheduler_setting["args"]
            )
        edges = Edges(criterions, gates=gates_list)

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
        trial=trial,
    )
    best_top1 = graph.train()
    return best_top1


if __name__ == "__main__":
    # Cteate study object
    study_name = f"bit_kd_{num_nodes}"
    optuna_dir = f"optuna/{study_name}"
    os.makedirs(optuna_dir, exist_ok=True)
    storage = JournalStorage(JournalFileStorage(os.path.join(optuna_dir, "optuna.log")))
    sampler = optuna.samplers.TPESampler()
    pruner = optuna.pruners.NopPruner()
    study = optuna.create_study(
        storage=storage,
        study_name=study_name,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,
    )
    # Start optimization
    study.optimize(objective, n_trials=n_trials)
