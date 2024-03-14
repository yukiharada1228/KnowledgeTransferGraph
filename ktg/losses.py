import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class SwAVLoss(nn.Module):
    # https://github.com/facebookresearch/swav/blob/main/main_swav.py#L289
    def __init__(
        self,
        batch_size,
        T=0.1,
        nmb_crops=[2, 6],
        crops_for_assign=[0, 1],
        sinkhorn_iterations=3,
        epsilon=0.05,
    ):
        super(SwAVLoss, self).__init__()
        # 損失関数
        self.batch_size = batch_size
        self.T = T
        self.nmb_crops = nmb_crops  # マルチクロップにより作成する画像の数
        self.crops_for_assign = (
            crops_for_assign  # クラスタ割り当てに使用するマルチクロップ画像の種類
        )
        self.sinkhorn_iterations = sinkhorn_iterations
        self.epsilon = epsilon

    @torch.no_grad()
    def distributed_sinkhorn(self, out, sinkhorn_iterations=3, epsilon=0.05):
        # https://github.com/facebookresearch/swav/blob/main/main_swav.py#L353
        Q = torch.exp(
            out / epsilon
        ).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1]  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for it in range(sinkhorn_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q.t()

    def forward(self, _z1, _z2, p1, _p2):
        # z1:特徴量, z2:None, p1:クラスタの確率分布（logits）, p2:None
        loss = 0
        for crop_id in self.crops_for_assign:
            # computing codes online
            with torch.no_grad():
                out = p1[
                    self.batch_size * crop_id : self.batch_size * (crop_id + 1)
                ].detach()  # 特徴量の取得
                q = self.distributed_sinkhorn(
                    out, self.sinkhorn_iterations, self.epsilon
                )[
                    -self.batch_size :
                ]  # クラスタ割り当て

            # cluster assignment prediction
            subloss = 0
            for v in np.delete(np.arange(np.sum(self.nmb_crops)), crop_id):
                x = p1[self.batch_size * v : self.batch_size * (v + 1)] / self.T
                subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
            loss += subloss / (np.sum(self.nmb_crops) - 1)
        loss /= len(self.crops_for_assign)
        return loss


class BarlowTwinsLoss(nn.Module):
    # https://github.com/facebookresearch/barlowtwins/blob/main/main.py#L211
    def __init__(self, batch_size, projector, lambd=0.0051):
        super(BarlowTwinsLoss, self).__init__()
        # 損失関数
        self.batch_size = batch_size
        self.lambd = lambd
        self.bn = nn.BatchNorm1d(projector.out_dim, affine=False)
        return

    def off_diagonal(self, x):
        # https://github.com/facebookresearch/barlowtwins/blob/main/main.py#L180
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2, p1, p2, **kwargs):
        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)  # バッチ正規化->内積->コサイン類似度
        c.div_(self.batch_size)

        # 損失を計算
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss
