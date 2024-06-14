# Import packages
import argparse
import os
from copy import deepcopy

import optuna
import torch
from optuna.storages import JournalFileStorage, JournalStorage
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from ktg import Edges, KnowledgeTransferGraph, Node, gates, losses
from ktg.models import cifar_models, projector, ssl_models
from ktg.transforms import ssl_transforms
from ktg.utils import (LARS, AverageMeter, KNNValidation, WorkerInitializer,
                       get_cosine_schedule_with_warmup, load_checkpoint,
                       set_seed)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42)
parser.add_argument("--num-nodes", default=7)
parser.add_argument("--n_trials", default=250)
parser.add_argument(
    "--models",
    default=["resnet18", "resnet34", "resnet50"],
)
parser.add_argument(
    "--gates",
    default=["ThroughGate", "CutoffGate"],
)
parser.add_argument(
    "--ssls",
    default=["SimCLR", "MoCo", "SimSiam", "BYOL", "SwAV", "BarlowTwins", "DINO"],
)
parser.add_argument(
    "--kds",
    default=["MSELoss", "KLLoss", "MSEKLLoss"],
)
parser.add_argument("--transforms", default="DINO")
parser.add_argument("--projector", default="SwAV")

args = parser.parse_args()
manual_seed = int(args.seed)
num_nodes = int(args.num_nodes)
n_trials = int(args.n_trials)
models_name = args.models
gates_name = args.gates
ssls_name = args.ssls
kds_name = args.kds
transforms_name = args.transforms
projector_name = args.projector

num_nodes = 7
study_name = f"dcl_{num_nodes}"
optuna_dir = f"../optuna/{study_name}"
storage = JournalStorage(JournalFileStorage(os.path.join(optuna_dir, "optuna.log")))
study = optuna.create_study(
    storage=storage,
    study_name=study_name,
    load_if_exists=True,
)

study_df = study.trials_dataframe()
complete_df = study_df[study_df.state == "COMPLETE"]
sorted_df = complete_df.sort_values(by="value", ascending=False)

top = 0
top_series = sorted_df.iloc[top]


def objective(trial):
    # Fix the seed value
    set_seed(manual_seed)

    # Prepare the CIFAR-10 for training
    accumulation_steps = 2**2
    batch_size = 512 // accumulation_steps
    num_workers = 10

    train_dataset = datasets.CIFAR10(root="data", train=True, download=True)
    val_dataset = datasets.CIFAR10(root="data", train=False, download=True)
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
            "lr": 0.3 * (batch_size * accumulation_steps / 256),
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
    for i in range(num_nodes):
        gates_list = []
        criterions = []
        for j in range(num_nodes):
            if i == j:
                loss_name = trial.suggest_categorical(f"{i}_{j}_loss", ["SSLLoss"])
            else:
                loss_name = trial.suggest_categorical(f"{i}_{j}_loss", ["MSEKLLoss"])
            criterions.append(getattr(losses, loss_name)())
            gate_name = trial.suggest_categorical(
                f"{i}_{j}_gate", [top_series[f"params_{i}_{j}_gate"]]
            )
            gates_list.append(getattr(gates, gate_name)(max_epoch))
        model_name = trial.suggest_categorical(
            f"{i}_model", [top_series[f"params_{i}_model"]]
        )
        ssl_name = trial.suggest_categorical(
            f"{i}_ssl", [top_series[f"params_{i}_ssl"]]
        )
        model = getattr(ssl_models, ssl_name)(
            encoder_func=getattr(cifar_models, model_name),
            batch_size=batch_size,
            projector_func=getattr(projector, f"{projector_name}Projector"),
        ).cuda()
        if (
            all(gate.__class__.__name__ == "CutoffGate" for gate in gates_list)
            and i != 0
        ):
            load_checkpoint(
                model=model,
                save_dir=f"checkpoint/pre-train/{model_name}/{projector_name}/{transforms_name}/{ssl_name}",
                is_best=True,
            )
        writer = SummaryWriter(
            f"runs/msekl_{num_nodes}/{projector_name}/{transforms_name}/{trial.number:04}/{i}_{model_name}_{ssl_name}"
        )
        save_dir = f"checkpoint/msekl_{num_nodes}/{projector_name}/{transforms_name}/{trial.number:04}/{i}_{model_name}_{ssl_name}"
        optimizer = LARS(model.parameters(), **optim_setting["args"])
        scheduler = get_cosine_schedule_with_warmup(
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
        accumulation_steps=accumulation_steps,
        trial=trial,
    )
    best_top1 = graph.train()
    return best_top1


if __name__ == "__main__":

    import os

    import optuna
    from optuna.storages import JournalFileStorage, JournalStorage

    # Cteate study object
    study_name = f"msekl_{num_nodes}"
    optuna_dir = f"optuna/{study_name}"
    os.makedirs(optuna_dir, exist_ok=True)
    storage = JournalStorage(JournalFileStorage(os.path.join(optuna_dir, "optuna.log")))
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
