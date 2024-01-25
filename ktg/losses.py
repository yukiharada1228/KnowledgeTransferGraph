import torch
import torch.nn as nn


class TotalLoss(nn.Module):
    def __init__(self, criterions, gates):
        super(TotalLoss, self).__init__()
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
                losses += [gate(criterion(target_output, label), epoch)]
            else:
                losses += [
                    gate(criterion(target_output, source_output.detach()), epoch)
                ]
        loss = torch.stack(losses).sum()
        return loss


class CrossEntropyLossSoftTarget(nn.Module):
    def __init__(self, T):
        super(CrossEntropyLossSoftTarget, self).__init__()
        self.T = T
        self.softmax = nn.Softmax(dim=-1)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_gt):
        y_pred_soft = y_pred / self.T
        y_gt_soft = y_gt / self.T
        return self.criterion(y_pred_soft, self.softmax(y_gt_soft)) * (self.T**2)


class KLDivLossSoftTarget(nn.Module):
    def __init__(self, T):
        super(KLDivLossSoftTarget, self).__init__()
        self.T = T
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, y_pred, y_gt):
        y_pred_soft = self.softmax(y_pred / self.T)
        y_gt_soft = self.softmax(y_gt / self.T)
        return self.kl_divergence(y_pred_soft, y_gt_soft) * (self.T**2)

    def kl_divergence(self, student, teacher):
        kl = teacher * torch.log((teacher / (student + 1e-10)) + 1e-10)
        kl = kl.sum(dim=1)
        loss = kl.mean()
        return loss
