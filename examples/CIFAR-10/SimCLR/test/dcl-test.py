import argparse
import os
from typing import List, Optional

import optuna
import torch
import torch.nn as nn
from optuna.storages import JournalFileStorage, JournalStorage
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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
    load_checkpoint,
)
from torchvision import transforms


def infer_model_names(
    best_trial: optuna.trial.FrozenTrial, num_nodes: int
) -> List[str]:
    """
    Optuna の trial 情報から各ノードのモデル名を復元する。
    """
    model_names: List[Optional[str]] = [None] * num_nodes

    # 1) params からの補完（ノード1以降）
    for i in range(1, num_nodes):
        if not model_names[i]:
            key = f"{i}_model"
            val = best_trial.params.get(key)
            if isinstance(val, str) and len(val) > 0:
                model_names[i] = val

    # 2) ノード0のフォールバック（学習既定: models[0] = resnet32）
    if not model_names[0]:
        model_names[0] = "resnet32"

    if any(m is None or len(m) == 0 for m in model_names):
        missing = [i for i, m in enumerate(model_names) if not m]
        raise RuntimeError(
            f"モデル名を特定できませんでした。trial={best_trial.number}, 欠損ノード={missing}"
        )
    return [str(m) for m in model_names]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-nodes", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--trial", type=int, default=None, help="固定したい試行番号")
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Optuna study 名。未指定時は dcl_{num_nodes} を使用",
    )
    parser.add_argument("--max-epoch", type=int, default=400)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--wd", type=float, default=1e-6)
    args = parser.parse_args()

    set_seed(args.seed)

    study_name = args.study_name or f"dcl_{args.num_nodes}"
    optuna_dir = os.path.join("../optuna", study_name)
    storage = JournalStorage(JournalFileStorage(os.path.join(optuna_dir, "optuna.log")))
    study = optuna.create_study(
        storage=storage, study_name=study_name, load_if_exists=True
    )

    if args.trial is not None:
        # 指定 trial を利用（存在チェック）
        frozen = None
        for t in study.trials:
            if t.number == args.trial:
                frozen = t
                break
        if frozen is None:
            raise ValueError(f"指定の trial が見つかりません: {args.trial}")
        best_trial = frozen
    else:
        # best_trial を採用
        if study.best_trial is None:
            raise RuntimeError(
                "best_trial が見つかりませんでした。学習が未完了の可能性があります。"
            )
        best_trial = study.best_trial

    model_names = infer_model_names(best_trial, args.num_nodes)

    # データセット: SimCLR 用に 2-view を生成。use_test_mode=True で train(=train+val) / test を得る
    base_train_ds, base_test_ds = get_datasets(use_test_mode=True)
    ssl_transform = SimCLRTransforms()
    train_dataset = CustomDataset(base_train_ds.subset, transform=ssl_transform)
    test_dataset = CustomDataset(base_test_ds, transform=ssl_transform)

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
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=WorkerInitializer(args.seed).worker_init_fn,
    )

    max_epoch = int(args.max_epoch)

    num_classes = 10
    nodes: List[Node] = []
    for i in range(args.num_nodes):
        # ゲートと損失
        gates_list = []
        criterions = []
        for j in range(args.num_nodes):
            if i == j:
                criterions.append(SimCLRLoss(args.batch_size))
            else:
                criterions.append(SimilarityMatrixKLDivLoss())
            gate_name = best_trial.params.get(f"{i}_{j}_gate")
            if gate_name is None:
                raise RuntimeError(
                    f"trial {best_trial.number:04} に {i}_{j}_gate が見つかりません"
                )
            gate = getattr(gates, gate_name)(max_epoch)
            gates_list.append(gate)

        all_cutoff = all(g.__class__.__name__ == "CutoffGate" for g in gates_list)

        model_name = model_names[i]

        def encoder_func(_mn=model_name):
            model = getattr(cifar_models, _mn)(num_classes)
            return model

        ssl_model = SimCLR(encoder_func).cuda()

        # 事前学習チェックポイントの読み込み（全入力Cutoffかつ i!=0 の場合）
        if all_cutoff and i != 0:
            pretrain_dir = os.path.join("checkpoint", "pre-train", model_name)
            load_checkpoint(model=ssl_model, save_dir=pretrain_dir, is_best=True)

        writer = SummaryWriter(
            f"runs/dcl_{args.num_nodes}/{best_trial.number:04}/{i}_{model_name}"
        )
        save_dir = (
            f"checkpoint/dcl_{args.num_nodes}/{best_trial.number:04}/{i}_{model_name}"
        )

        # Optimizer/Scheduler（LARS + Warmup Cosine）
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
            num_training_steps=max_epoch,
            num_cycles=0.5,
        )

        edges = [Edge(c, g) for c, g in zip(criterions, gates_list)]

        def knn_eval_fn():
            # get_datasets を使い、Normalize なしの前処理に差し替え
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
            save_dir=save_dir,
            optimizer=optimizer,
            scheduler=scheduler,
            edges=edges,
            loss_meter=AverageMeter(),
            score_meter=AverageMeter(),
            eval=knn_eval_fn,
        )
        nodes.append(node)

    graph = KnowledgeTransferGraph(
        nodes=nodes,
        max_epoch=max_epoch,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        trial=None,
    )

    best_score = graph.train()
    print("-")
    print(f"Best trial = {best_trial.number:04}")
    print(f"Node 0 (primary) best KNN top1 = {best_score:.2f}%")


if __name__ == "__main__":
    main()
