# Import packages
import argparse
import os
from copy import deepcopy

import optuna
import torch
from optuna.storages import JournalFileStorage, JournalStorage
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from ktg import Edges, KnowledgeTransferGraph, Node, gates, losses
from ktg.dataset.tinyimagenet import TinyImageNet
from ktg.models import cifar_models, projector, ssl_models
from ktg.transforms import ssl_transforms
from ktg.utils import (AverageMeter, KNNValidation, WorkerInitializer, load_checkpoint, set_seed)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42)
parser.add_argument("--num-nodes", default=2)
parser.add_argument("--models", default=["resnet18"])
parser.add_argument("--gates", default=["PositiveGammaGate", "NegativeGammaGate"])
parser.add_argument(
    "--ssls",
    default=["SimCLR", 
             "MoCo", 
             "SimSiam", 
             "BYOL", 
             "SwAV", 
             "BarlowTwins", 
             "DINO"]
)
parser.add_argument("--kds", default=["KLLoss"])
parser.add_argument("--transforms", default="DINO")
parser.add_argument("--accumulation_steps", default=2)

args = parser.parse_args()
manual_seed = int(args.seed)
num_nodes = int(args.num_nodes)
models_name = args.models
gates_name = args.gates
ssls_name = args.ssls
kds_name = args.kds
transforms_name = args.transforms
accumulation_steps = int(args.accumulation_steps)


def objective(trial):
    # Fix the seed value
    set_seed(manual_seed)

    batch_size = 256 // accumulation_steps
    num_workers = 10

    train_dataset = TinyImageNet('tiny-imagenet-200', split='train')
    val_dataset = TinyImageNet('tiny-imagenet-200', split='val')
    transform = getattr(ssl_transforms, f"{transforms_name}Transforms")(size_crops=[64, 32])
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
        "name": "AdamW",
        "args": {
            "lr": 3e-4 * (batch_size * accumulation_steps / 256),
            "weight_decay": 1e-6,
        },
    }

    nodes = []
    for i in range(num_nodes):
        gates_list = []
        criterions = []
        for j in range(num_nodes):
            if i == j:
                loss_name = trial.suggest_categorical(f"{i}_{j}_loss", ["SSLLoss"])
                criterions.append(getattr(losses, loss_name)())
                gate_name = trial.suggest_categorical(
                    f"{i}_{j}_gate", ["ThroughGate"]
                )
                gates_list.append(getattr(gates, gate_name)(max_epoch))
            else:
                loss_name = trial.suggest_categorical(f"{i}_{j}_loss", ["KLLoss"])
                criterions.append(getattr(losses, loss_name)(lam=100))
                gamma = trial.suggest_float(f"{i}_{j}_gamma", 0.01, 100, log=True)
                gate_name = trial.suggest_categorical(
                    f"{i}_{j}_gate", gates_name
                )
                gates_list.append(getattr(gates, gate_name)(max_epoch, gamma))
        model_name = trial.suggest_categorical(
            f"{i}_model", models_name
        )
        ssl_name = trial.suggest_categorical(
            f"{i}_ssl", ssls_name
        )
        model = getattr(ssl_models, ssl_name)(
            encoder_func=getattr(cifar_models, model_name),
            batch_size=batch_size,
            projector_func=getattr(projector, f"{ssl_name}Projector"),
        ).cuda()
        if (
            all(gate.__class__.__name__ == "CutoffGate" for gate in gates_list)
            and i != 0
        ):
            load_checkpoint(
                model=model,
                save_dir=f"checkpoint/pre-train/{model_name}/{ssl_name}/{transforms_name}/{ssl_name}",
                is_best=True,
            )
        writer = SummaryWriter(
            f"runs/dcl_{num_nodes}/{ssl_name}/{transforms_name}/{trial.number:04}/{i}_{model_name}_{ssl_name}"
        )
        save_dir = f"checkpoint/dcl_{num_nodes}/{ssl_name}/{transforms_name}/{trial.number:04}/{i}_{model_name}_{ssl_name}"
        optimizer = getattr(torch.optim, optim_setting["name"])(
            model.parameters(), **optim_setting["args"]
        )
        edges = Edges(criterions, gates=gates_list)

        node = Node(
            model=model,
            writer=writer,
            scaler=torch.cuda.amp.GradScaler(),
            save_dir=save_dir,
            optimizer=optimizer,
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
    study_name = f"dcl_{num_nodes}"
    optuna_dir = f"optuna/{study_name}"
    os.makedirs(optuna_dir, exist_ok=True)
    storage = JournalStorage(JournalFileStorage(os.path.join(optuna_dir, "optuna.log")))
    sampler = optuna.samplers.TPESampler(multivariate=True)
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
    study.optimize(objective)
