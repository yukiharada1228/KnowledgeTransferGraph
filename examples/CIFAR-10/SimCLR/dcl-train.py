import argparse
import os

import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from optuna.storages import JournalFileStorage, JournalStorage

from ktg import Edge, KnowledgeTransferGraph, Node, gates
from ktg.losses import SimCLRLoss, SimilarityMatrixKLDivLoss
from ktg.models.ssl_models import SimCLR
from ktg.models import cifar_models
from ktg.transforms.ssl_transforms import SimCLRTransforms
from ktg.dataset.cifar_datasets.cifar10 import get_datasets
from ktg.dataset.custom_dataset import CustomDataset
from ktg.utils import (
    AverageMeter,
    WorkerInitializer,
    set_seed,
    LARS,
    get_cosine_schedule_with_warmup,
    KNNValidation,
)
from torchvision import transforms


## ラッパー不要: SimilarityMatrixKLDivLoss 側で [z1,z2] / [loss,z1,z2] の双方に対応


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-nodes", type=int, default=3)
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument(
        "--models",
        nargs="*",
        type=str,
        default=["resnet32", "resnet110", "wideresnet28_2"],
        help="使用する CIFAR モデル名の候補。ノード数ぶん順に割り当て（不足時は最後を繰り返し）。",
    )
    parser.add_argument(
        "--gates",
        nargs="*",
        type=str,
        default=[
            "ThroughGate",
            "CutoffGate",
            "PositiveLinearGate",
            "NegativeLinearGate",
        ],
        help="探索対象のゲート候補（ktg.gates 内のクラス名）",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--max-epoch", type=int, default=400)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--wd", type=float, default=1e-6)
    args = parser.parse_args()

    def objective(trial: optuna.Trial):
        # 乱数固定
        set_seed(args.seed)

        # データセット（train/val を使用、SimCLR 用の 2 view を生成）
        base_train_ds, base_val_ds = get_datasets()
        ssl_transform = SimCLRTransforms()
        train_dataset = CustomDataset(base_train_ds.subset, transform=ssl_transform)
        val_dataset = CustomDataset(base_val_ds.subset, transform=ssl_transform)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=WorkerInitializer(args.seed).worker_init_fn,
        )
        test_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=WorkerInitializer(args.seed).worker_init_fn,
        )

        # ノード群の構築
        num_classes = 10
        num_nodes = args.num_nodes
        models_name = args.models
        gates_name = args.gates

        nodes = []
        for i in range(num_nodes):
            # モデル定義（SimCLR）
            model_name = models_name[i] if i < len(models_name) else models_name[-1]

            def encoder_func(_mn=model_name):
                model = getattr(cifar_models, _mn)(num_classes)
                return model

            ssl_model = SimCLR(encoder_func).cuda()

            # Optimizer/Scheduler（SimCLR 推奨設定に合わせて LARS + Cosine）
            optimizer = LARS(
                ssl_model.parameters(),
                lr=args.lr,
                weight_decay=args.wd,
                momentum=0.9,
                eta=0.001,
                weight_decay_filter=False,
                lars_adaptation_filter=False,
            )
            scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=args.warmup_epochs,
                num_training_steps=args.max_epoch,
                num_cycles=0.5,
            )
            edges_list = []
            for j in range(num_nodes):
                if i == j:
                    criterion = SimCLRLoss(args.batch_size)
                else:
                    criterion = SimilarityMatrixKLDivLoss()
                gate_name = trial.suggest_categorical(f"{i}_{j}_gate", gates_name)
                gate = getattr(gates, gate_name)(args.max_epoch)
                edges_list.append(Edge(criterion, gate))

            writer = SummaryWriter(
                f"runs/dcl_{args.num_nodes}/{trial.number:04}/{i}_{model_name}"
            )
            save_dir = (
                f"checkpoint/dcl_{args.num_nodes}/{trial.number:04}/{i}_{model_name}"
            )

            # pre-train.py と同様の KNN 評価関数をローカルに定義
            def knn_eval_fn():
                le_train_ds, le_test_ds = get_datasets(use_test_mode=True)
                transform = transforms.Compose([transforms.ToTensor()])
                le_train_ds.transform = transform
                le_test_ds.transform = transform
                evaluator = KNNValidation(
                    model=ssl_model,
                    train_dataset=le_train_ds,
                    test_dataset=le_test_ds,
                    K=20,
                )
                return float(evaluator().item())

            node = Node(
                model=ssl_model,
                writer=writer,
                scaler=torch.cuda.amp.GradScaler(),
                optimizer=optimizer,
                scheduler=scheduler,
                edges=edges_list,
                loss_meter=AverageMeter(),
                score_meter=AverageMeter(),
                eval=knn_eval_fn,
                save_dir=save_dir,
            )
            nodes.append(node)

        graph = KnowledgeTransferGraph(
            nodes=nodes,
            max_epoch=args.max_epoch,
            train_dataloader=train_loader,
            test_dataloader=test_loader,
            trial=trial,
        )
        best_score = graph.train()
        return best_score

    # Optuna Study 設定と最適化開始
    study_name = f"dcl_{args.num_nodes}"
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
    study.optimize(objective, n_trials=args.n_trials)


if __name__ == "__main__":
    main()
