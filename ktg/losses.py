import torch
import torch.nn as nn


class CrossEntropySoftTargetLoss(nn.Module):
    def __init__(self, T):
        super(CrossEntropySoftTargetLoss, self).__init__()
        self.T = T
        self.softmax = nn.Softmax(dim=-1)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_gt):
        y_pred_soft = y_pred / self.T
        y_gt_soft = y_gt / self.T
        return self.criterion(y_pred_soft, self.softmax(y_gt_soft)) * (self.T**2)


class KLDivLoss(nn.Module):
    def __init__(self):
        super(KLDivLoss, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, y_pred, y_gt):
        y_pred_soft = self.softmax(y_pred)
        y_gt_soft = self.softmax(y_gt)
        return self.kl_divergence(y_pred_soft, y_gt_soft)

    def kl_divergence(self, student, teacher):
        kl = teacher * torch.log((teacher / (student + 1e-10)) + 1e-10)
        kl = kl.sum(dim=1)
        loss = kl.mean()
        return loss


class SSLLoss(nn.Module):
    def __init__(self):
        super(SSLLoss, self).__init__()

    def forward(self, target_output, _):
        loss = target_output[0]
        return loss


class SimSiamLoss(nn.Module):
    # https://github.com/facebookresearch/simsiam/blob/main/main_simsiam.py#L295
    def __init__(self):
        super(SimSiamLoss, self).__init__()
        # コサイン類似度
        self.criterion = nn.CosineSimilarity(dim=1)

    def forward(self, z1, z2, p1, p2):
        # 負のコサイン類似度の計算
        loss = -(self.criterion(p1, z2).mean() + self.criterion(p2, z1).mean()) * 0.5
        return loss
