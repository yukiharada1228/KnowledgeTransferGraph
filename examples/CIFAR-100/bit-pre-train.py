# Import packages
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ktg import Edges, KnowledgeTransferGraph, Node
from ktg.dataset.cifar_datasets.cifar100 import get_datasets
from ktg.gates import ThroughGate
from ktg.models import cifar_models
from ktg.utils import AverageMeter, WorkerInitializer, set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42)
parser.add_argument(
    "--models", default=["bit_resnet32_b158", "bit_resnet56_b158", "bit_resnet110_b158"]
)
parser.add_argument("--n_trials", default=100)

args = parser.parse_args()
n_trials = int(args.n_trials)
manualSeed = int(args.seed)
models_name = args.models


def objective(trial):
    # Fix the seed value
    set_seed(manualSeed)

    # lr = trial.suggest_float('lr', 0.001, 1.0, log=True)
    lr = 0.45422698806566725
    # wd = trial.suggest_float('wd', 5e-6, 5.0, log=True)
    wd = 7.310284744110568e-05

    # Prepare the CIFAR-100 for training
    batch_size = 64
    num_workers = 10

    train_dataset, val_dataset, _ = get_datasets()

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
    # max_epoch = trial.suggest_int('epoch', 200, 1000)

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

    num_classes = 100
    nodes = []

    gates = [ThroughGate(max_epoch)]
    criterions = [nn.CrossEntropyLoss()]
    model_name = trial.suggest_categorical("model", models_name)
    model = getattr(cifar_models, model_name)(num_classes).cuda()
    writer = SummaryWriter(f"runs/bit_pre_train/{trial.number:04}/{model_name}")
    save_dir = f"checkpoint/bit_pre_train/{trial.number:04}/{model_name}"
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

    # start_rate = trial.suggest_int('start_rate', 1, 10)
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
        trial=trial,
    )
    best_top1 = graph.train()
    return best_top1


if __name__ == "__main__":
    import os

    import optuna
    from optuna.storages import JournalFileStorage, JournalStorage

    # Cteate study object
    study_name = f"bit_pretrain"
    optuna_dir = f"optuna/{study_name}"
    os.makedirs(optuna_dir, exist_ok=True)
    storage = JournalStorage(JournalFileStorage(os.path.join(optuna_dir, "optuna.log")))
    # sampler = optuna.samplers.TPESampler(multivariate=True)
    sampler = optuna.samplers.BruteForceSampler()
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
