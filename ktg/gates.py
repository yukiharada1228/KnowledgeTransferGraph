import torch
import torch.nn as nn


class ThroughGate(nn.Module):
    def __init__(self, max_epoch):
        super(ThroughGate, self).__init__()

    def forward(self, loss, epoch):
        return loss


class CutoffGate(nn.Module):
    def __init__(self, max_epoch):
        super(CutoffGate, self).__init__()

    def forward(self, loss, epoch):
        loss = torch.zeros_like(loss, requires_grad=True)
        return loss


class PositiveLinearGate(nn.Module):
    def __init__(self, max_epoch):
        super(PositiveLinearGate, self).__init__()
        self.max_epoch = max_epoch

    def forward(self, loss, epoch):
        loss_weight = epoch / self.max_epoch
        loss *= loss_weight
        return loss


class NegativeLinearGate(nn.Module):
    def __init__(self, max_epoch):
        super(NegativeLinearGate, self).__init__()
        self.max_epoch = max_epoch

    def forward(self, loss, epoch):
        loss_weight = (self.max_epoch - epoch) / self.max_epoch
        loss *= loss_weight
        return loss


class PositiveGammaGate(nn.Module):
    def __init__(self, max_epoch, gamma):
        super(PositiveGammaGate, self).__init__()
        self.max_epoch = max_epoch
        self.gamma = gamma

    def forward(self, loss, epoch):
        loss_weight = epoch / self.max_epoch
        loss *= loss_weight ** (1 / self.gamma)
        return loss


class NegativeGammaGate(nn.Module):
    def __init__(self, max_epoch, gamma):
        super(NegativeGammaGate, self).__init__()
        self.max_epoch = max_epoch
        self.gamma = gamma

    def forward(self, loss, epoch):
        loss_weight = (self.max_epoch - epoch) / self.max_epoch
        loss *= loss_weight ** (1 / self.gamma)
        return loss
