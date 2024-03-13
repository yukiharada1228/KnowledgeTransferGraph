import os
import time
from dataclasses import dataclass

import optuna
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ktg.utils import AverageMeter, accuracy, save_checkpoint


class Edges(nn.Module):
    def __init__(self, criterions, gates):
        super(Edges, self).__init__()
        self.criterions = criterions
        self.gates = gates

    def forward(self, model_id, outputs, labels, epoch):
        if model_id < 0 or model_id >= len(outputs):
            raise ValueError(f"Invalid model_id: {model_id}")
        losses = []
        target_output = outputs[model_id]
        label = labels[model_id]
        for i, (source_output, criterion, gate) in enumerate(
            zip(outputs, self.criterions, self.gates)
        ):
            if i == model_id:
                loss = gate(criterion(target_output, label), epoch)
            elif gate.__class__.__name__ != "CutoffGate":
                losses += [
                    gate(criterion(target_output, source_output.detach()), epoch)
                ]
        if len(losses) > 0:
            loss = loss + torch.stack(losses).mean()
        return loss


@dataclass
class Node:
    model: nn.Module
    writer: SummaryWriter
    scaler: torch.cuda.amp.GradScaler
    save_dir: str
    optimizer: Optimizer
    scheduler: LRScheduler
    edges: Edges
    loss_meter: AverageMeter
    top1_meter: AverageMeter
    best_top1: float = 0.0


class KnowledgeTransferGraph:
    def __init__(
        self,
        nodes: list[Node],
        max_epoch: int,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        trial=None,
    ):
        print("Welcome to KTG!!!")
        self.nodes = nodes
        for node in nodes:
            os.makedirs(node.save_dir, exist_ok=True)
        self.max_epoch = max_epoch
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.trial = trial

    def train_on_batch(self, image, label, epoch):
        if type(image) == list:
            image = [img.cuda() for img in image]
        else:
            image = image.cuda()
        label = label.cuda()

        outputs = []
        labels = []
        for node in self.nodes:
            node.model.train()
            with torch.cuda.amp.autocast():
                y = node.model(image)
            outputs.append(y)
            labels.append(label)

        for model_id, node in enumerate(self.nodes):
            with torch.cuda.amp.autocast():
                loss = node.edges(model_id, outputs, labels, epoch)
            if loss > 0:
                node.scaler.scale(loss).backward()
                node.scaler.step(node.optimizer)
                node.optimizer.zero_grad()
                node.scaler.update()
            if type(image) == torch.Tensor:
                [top1] = node.eval(outputs[model_id], labels[model_id], topk=(1,))
                node.top1_meter.update(top1.item(), labels[model_id].size(0))
            node.loss_meter.update(loss.item(), labels[model_id].size(0))

    def test_on_batch(self, image, label):
        if type(image) == torch.Tensor:
            image = image.cuda()
            label = label.cuda()

            outputs = []
            labels = []
            for node in self.nodes:
                node.model.eval()
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        y = node.model(image)
                outputs.append(y)
                labels.append(label)

            for model_id, node in enumerate(self.nodes):
                [top1] = accuracy(outputs[model_id], labels[model_id], topk=(1,))
                node.top1_meter.update(top1.item(), labels[model_id].size(0))

    def train(self):
        for epoch in range(1, self.max_epoch + 1):
            print("epoch %d" % epoch)
            start_time = time.time()

            for image, label in self.train_dataloader:
                self.train_on_batch(image=image, label=label, epoch=epoch - 1)
            for model_id, node in enumerate(self.nodes):
                train_lr = node.optimizer.param_groups[0]["lr"]
                train_loss = node.loss_meter.avg
                train_top1 = node.top1_meter.avg
                node.writer.add_scalar("train_lr", train_lr, epoch)
                node.writer.add_scalar("train_loss", train_loss, epoch)
                node.writer.add_scalar("train_top1", train_top1, epoch)
                node.scheduler.step()
                print(
                    "model_id: {0:}   loss :train={1:.3f}   top1 :train={2:.3f}".format(
                        model_id, train_loss, train_top1
                    )
                )
                node.loss_meter.reset()
                node.top1_meter.reset()

            for image, label in self.test_dataloader:
                self.test_on_batch(image=image, label=label)
            for model_id, node in enumerate(self.nodes):
                test_top1 = node.top1_meter.avg
                node.writer.add_scalar("test_top1", test_top1, epoch)
                print("model_id: {0:}   top1 :test={1:.3f}".format(model_id, test_top1))
                if node.best_top1 <= node.top1_meter.avg:
                    save_checkpoint(node.model, node.save_dir, epoch, is_best=True)
                    node.best_top1 = node.top1_meter.avg
                if model_id == 0 and self.trial is not None:
                    self.trial.report(test_top1, step=epoch)
                    if self.trial.should_prune():
                        raise optuna.TrialPruned()
                node.top1_meter.reset()
            elapsed_time = time.time() - start_time
            print("  elapsed_time:{0:.3f}[sec]".format(elapsed_time))

        for node in self.nodes:
            node.writer.close()

        best_top1 = self.nodes[0].best_top1
        return best_top1

    def __len__(self):
        return len(self.nodes)
