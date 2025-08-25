# Import packages
import argparse
import math
import os

import optuna
import torch
import torch.nn as nn
from optuna.storages import JournalFileStorage, JournalStorage
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ktg import Edge, KnowledgeTransferGraph, Node, gates
from ktg.dataset.cifar_datasets.cifar100 import get_datasets
from ktg.losses import KLDivLoss
from ktg.models import cifar_models
from ktg.utils import AverageMeter, WorkerInitializer, load_checkpoint, set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42)
parser.add_argument("--num-nodes", default=3)
parser.add_argument("--n_trials", default=100)
parser.add_argument(
    "--models",
    default=["resnet32", "resnet110", "wideresnet28_2"],
    nargs="*",
    type=str,
)
# SinGate の探索範囲
parser.add_argument("--num-cycles-min", type=float, default=0.0)
parser.add_argument("--num-cycles-max", type=float, default=2.0)
parser.add_argument("--phase-min", type=float, default=-math.pi)
parser.add_argument("--phase-max", type=float, default=math.pi)

args = parser.parse_args()
manualSeed = int(args.seed)
num_nodes = int(args.num_nodes)
n_trials = int(args.n_trials)
models_name = args.models
num_cycles_min = float(args.num_cycles_min)
num_cycles_max = float(args.num_cycles_max)
phase_min = float(args.phase_min)
phase_max = float(args.phase_max)


def objective(trial):
    # Fix the seed value
    set_seed(manualSeed)

    # Prepare the CIFAR-100 for training
    batch_size = 64
    num_workers = 10

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
    nodes = []
    for i in range(num_nodes):
        gates_list = []
        criterions = []
        for j in range(num_nodes):
            if i == j:
                criterions.append(nn.CrossEntropyLoss())
            else:
                criterions.append(KLDivLoss())
            # SinGate のパラメータを各エッジで探索
            num_cycles = trial.suggest_float(
                f"{i}_{j}_num_cycles", num_cycles_min, num_cycles_max
            )
            phase = trial.suggest_float(f"{i}_{j}_phase", phase_min, phase_max)
            gate = gates.SinGate(max_epoch, num_cycles=num_cycles, phase=phase)
            gates_list.append(gate)
        if i == 0:
            model_name = models_name[0]
        else:
            model_name = trial.suggest_categorical(f"{i}_model", models_name)
        model = getattr(cifar_models, model_name)(num_classes).cuda()
        if (
            all(gate.__class__.__name__ == "CutoffGate" for gate in gates_list)
            and i != 0
        ):
            load_checkpoint(
                model=model, save_dir=f"checkpoint/pre-train/{model_name}", is_best=True
            )
        writer = SummaryWriter(
            f"runs/dcl_sin_{num_nodes}/{trial.number:04}/{i}_{model_name}"
        )
        optimizer = getattr(torch.optim, optim_setting["name"])(
            model.parameters(), **optim_setting["args"]
        )
        scheduler = getattr(torch.optim.lr_scheduler, scheduler_setting["name"])(
            optimizer, **scheduler_setting["args"]
        )
        edges = [Edge(c, g) for c, g in zip(criterions, gates_list)]

        node = Node(
            model=model,
            writer=writer,
            scaler=torch.cuda.amp.GradScaler(),
            optimizer=optimizer,
            scheduler=scheduler,
            edges=edges,
            loss_meter=AverageMeter(),
            score_meter=AverageMeter(),
        )
        nodes.append(node)

    graph = KnowledgeTransferGraph(
        nodes=nodes,
        max_epoch=max_epoch,
        train_dataloader=train_dataloader,
        test_dataloader=val_dataloader,
        trial=trial,
    )
    best_score = graph.train()
    return best_score


if __name__ == "__main__":
    # Cteate study object
    study_name = f"dcl_sin_{num_nodes}"
    optuna_dir = f"optuna/{study_name}"
    os.makedirs(optuna_dir, exist_ok=True)
    storage = JournalStorage(JournalFileStorage(os.path.join(optuna_dir, "optuna.log")))
    sampler = optuna.samplers.TPESampler(multivariate=True)
    pruner = optuna.pruners.HyperbandPruner()
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
