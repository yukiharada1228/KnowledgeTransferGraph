# Import packages
import argparse
import os
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from ktg import Edges, KnowledgeTransferGraph, Node, gates, losses
from ktg.dataset.tinyimagenet import TinyImageNet
from ktg.models import cifar_models, projector, ssl_models
from ktg.transforms import ssl_transforms
from ktg.utils import AverageMeter, KNNValidation, WorkerInitializer, set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42)
parser.add_argument("--num-nodes", default=2)
parser.add_argument("--model", default="resnet18")
parser.add_argument("--ssl", default="SwAV")
parser.add_argument(
    "--ssls",
    default=["SimCLR", "MoCo", "SimSiam", "BYOL", "SwAV", "BarlowTwins"],
)
parser.add_argument("--kd", default="KLLoss")
parser.add_argument("--transforms", default="DINO")
parser.add_argument("--accumulation_steps", default=1)

args = parser.parse_args()
manual_seed = int(args.seed)
num_nodes = int(args.num_nodes)
model_name = args.model
ssl_name = args.ssl
ssls_name = args.ssls
kd_name = args.kd
transforms_name = args.transforms
accumulation_steps = int(args.accumulation_steps)


def objective(trial):
    # Fix the seed value
    set_seed(manual_seed)
    batch_size = 256 // accumulation_steps
    num_workers = 10
    train_dataset = TinyImageNet("tiny-imagenet-200", split="train")
    val_dataset = TinyImageNet("tiny-imagenet-200", split="val")
    transform = getattr(ssl_transforms, f"{transforms_name}Transforms")(
        size_crops=[64, 32]
    )
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
                criterions.append(getattr(losses, "SSLLoss")())
            else:
                criterions.append(getattr(losses, kd_name)(T=0.1, lam=1))
            gates_list.append(getattr(gates, "ThroughGate")(max_epoch))
        if i == 0:
            ssl = ssl_name
        else:
            ssl = trial.suggest_categorical(f"{i}_ssl", ssls_name)
        model = getattr(ssl_models, ssl)(
            encoder_func=getattr(cifar_models, model_name),
            batch_size=batch_size,
            projector_func=getattr(projector, f"{ssl_name}Projector"),
        ).cuda()
        writer = SummaryWriter(
            f"runs/dml_{num_nodes}_{ssl_name}/{transforms_name}/{trial.number:04}/{i}_{model_name}_{ssl}"
        )
        save_dir = f"checkpoint/dml_{num_nodes}_{ssl_name}/{transforms_name}/{trial.number:04}/{i}_{model_name}_{ssl}"
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
    import optuna
    from optuna.storages import JournalFileStorage, JournalStorage

    study_name = f"dml_{num_nodes}_{ssl_name}"
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
    study.optimize(objective)
