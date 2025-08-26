import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ktg import KnowledgeTransferGraph, Node, build_edges
from ktg.gates import ThroughGate
from ktg.models.ssl_models import SimCLR
from ktg.transforms.ssl_transforms import SimCLRTransforms
from ktg.utils import (
    AverageMeter,
    WorkerInitializer,
    set_seed,
    LARS,
    get_cosine_schedule_with_warmup,
    KNNValidation,
)
from ktg.losses import SimCLRLoss
from ktg.dataset.cifar_datasets.cifar10 import get_datasets
from ktg.dataset.custom_dataset import CustomDataset
from ktg.models import cifar_models
from torchvision import datasets, transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42)
    parser.add_argument("--lr", default=1.0)
    parser.add_argument("--wd", default=1e-6)
    parser.add_argument("--model", default="resnet32")
    args = parser.parse_args()

    manualSeed = int(args.seed)
    lr = float(args.lr)
    wd = float(args.wd)
    model_name = args.model

    set_seed(manualSeed)

    base_train_ds, base_test_ds = get_datasets(use_test_mode=True)
    ssl_transform = SimCLRTransforms()
    train_dataset = CustomDataset(base_train_ds.subset, transform=ssl_transform)
    base_test_ds.transform = None
    test_dataset = CustomDataset(base_test_ds, transform=ssl_transform)

    batch_size = 512
    num_workers = 10
    max_epoch = 400
    warmup_epochs = 10

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=WorkerInitializer(manualSeed).worker_init_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=WorkerInitializer(manualSeed).worker_init_fn,
    )

    def encoder_func():
        model = getattr(cifar_models, model_name)(10)
        return model

    ssl_model = SimCLR(encoder_func).cuda()

    # 自己教師あり学習損失のみを使用
    gates_list = [ThroughGate(max_epoch)]

    writer = SummaryWriter(f"runs/pre-train/{model_name}")
    save_dir = f"checkpoint/pre-train/{model_name}"

    optimizer = LARS(
        ssl_model.parameters(),
        lr=lr,
        weight_decay=wd,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=False,
        lars_adaptation_filter=False,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_epochs,
        num_training_steps=max_epoch,
        num_cycles=0.5,
    )

    edges = build_edges([SimCLRLoss(batch_size)], gates_list)

    def knn_eval_fn(_model=ssl_model):
        # get_datasets を使い、Normalize なしの前処理に差し替え
        le_train_ds, le_test_ds = get_datasets(use_test_mode=True)

        transform = transforms.Compose([transforms.ToTensor()])

        # transform を上書き
        le_train_ds.transform = transform
        le_test_ds.transform = transform

        evaluator = KNNValidation(
            model=_model,
            train_dataset=le_train_ds,
            test_dataset=le_test_ds,
            K=20,
        )
        # tensor -> float
        return float(evaluator().item())

    node = Node(
        model=ssl_model,
        writer=writer,
        scaler=torch.cuda.amp.GradScaler(),
        save_dir=save_dir,
        optimizer=optimizer,
        scheduler=scheduler,
        edges=edges,
        loss_meter=AverageMeter(),
        score_meter=AverageMeter(),
        eval=knn_eval_fn,
    )

    graph = KnowledgeTransferGraph(
        nodes=[node],
        max_epoch=max_epoch,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
    )
    graph.train()


if __name__ == "__main__":
    main()
