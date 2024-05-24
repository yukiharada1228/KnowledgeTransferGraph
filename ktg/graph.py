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
                losses.append(gate(criterion(target_output, label), epoch))
            else:
                losses.append(gate(criterion(target_output, source_output), epoch))
        loss = torch.stack(losses).sum()
        return loss


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


@dataclass
class Node:
    model: nn.Module
    writer: SummaryWriter
    scaler: torch.cuda.amp.GradScaler
    save_dir: str
    optimizer: Optimizer
    edges: Edges
    loss_meter: AverageMeter
    top1_meter: AverageMeter
    scheduler: LRScheduler = None
    wdscheduler: WeightDecayScheduler = None
    best_top1: float = 0.0
    eval: nn.Module = accuracy


class KnowledgeTransferGraph:
    def __init__(
        self,
        nodes: list[Node],
        max_epoch: int,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        accumulation_steps: int = 1,
        trial=None,
    ):
        print("Welcome to KTG!!!")
        self.nodes = nodes
        for node in nodes:
            os.makedirs(node.save_dir, exist_ok=True)
        self.max_epoch = max_epoch
        self.accumulation_steps = accumulation_steps
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.trial = trial
        self.data_length = len(self.train_dataloader)

    def train_on_batch(self, image, label, epoch, num_iter):
        if type(image) == list:
            if len(image) == 2:
                image = [img.cuda() for img in image]
                image.append(None)
            elif len(image) == 3:
                image = [
                    image[0].cuda(),
                    image[1].cuda(),
                    [img.cuda() for img in image[2]],
                ]
            else:
                raise Exception("Invalid image list length. Expected length 2 or 3.")
        else:
            image = image.cuda()
        label = label.cuda()

        outputs = []
        labels = []
        for node in self.nodes:
            node.model.train()
            with torch.cuda.amp.autocast():
                if type(image) == list:
                    y = node.model(image[0], image[1], image[2])
                else:
                    y = node.model(image)
            outputs.append(y)
            labels.append(label)

        for model_id, node in enumerate(self.nodes):
            with torch.cuda.amp.autocast():
                loss = node.edges(model_id, outputs, labels, epoch)
                if loss != 0:
                    node.scaler.scale(loss / self.accumulation_steps).backward()
                    if ((num_iter + 1) % self.accumulation_steps == 0) or (
                        (num_iter + 1) == self.data_length
                    ):
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
                [top1] = node.eval(outputs[model_id], labels[model_id], topk=(1,))
                node.top1_meter.update(top1.item(), labels[model_id].size(0))

    def train(self):
        for epoch in range(1, self.max_epoch + 1):
            print("epoch %d" % epoch)
            start_time = time.time()

            for idx, (image, label) in enumerate(self.train_dataloader):
                self.train_on_batch(
                    image=image, label=label, epoch=epoch - 1, num_iter=idx
                )
            for model_id, node in enumerate(self.nodes):
                train_lr = node.optimizer.param_groups[0]["lr"]
                train_loss = node.loss_meter.avg
                train_top1 = node.top1_meter.avg
                node.writer.add_scalar("train_lr", train_lr, epoch)
                node.writer.add_scalar("train_loss", train_loss, epoch)
                node.writer.add_scalar("train_top1", train_top1, epoch)
                if node.scheduler is not None:
                    node.scheduler.step()
                if node.wdscheduler is not None:
                    node.wdscheduler.step(epoch)
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
                if node.top1_meter.avg == 0.0:
                    node.top1_meter.avg = node.eval()
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
