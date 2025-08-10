import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class KLDivLoss(nn.Module):
    def __init__(self, T=1):
        super(KLDivLoss, self).__init__()
        self.T = T
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, y_pred, y_gt):
        y_pred_soft = self.softmax(y_pred / self.T)
        y_gt_soft = self.softmax(y_gt.detach() / self.T)
        # 温度スケーリングに伴う勾配スケール補正 T^2
        return (self.T**2) * self.kl_divergence(y_pred_soft, y_gt_soft)

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


class SimCLRLoss(nn.Module):
    # 有志の方のPyTorch実装版のSimCLRのプログラムを参考に作成
    def __init__(self, batch_size, T=0.3):
        super(SimCLRLoss, self).__init__()
        # NT-Xent損失
        self.batch_size = batch_size
        self.N = 2 * batch_size  # 1回のパラメータ更新で使用するサンプル数
        self.T = T  # 温度パラメータ
        self.mask = self.mask_correlated_samples(
            batch_size
        )  # ポジティブ/ネガティブペアの類似度を取り出すためのマスク
        self.criterion = nn.CrossEntropyLoss(
            reduction="sum"
        )  # softmaxを内包したクロスエントロピー

    def mask_correlated_samples(self, batch_size):
        # 同一特徴量間の類似度とポジティブペアの類似度を削除するためのマスクの作成
        mask = torch.ones((self.N, self.N), dtype=bool)
        mask = mask.fill_diagonal_(0)  # 同一特徴量間の類似度が入っている位置の値を0に
        for i in range(batch_size):
            mask[i, batch_size + i] = (
                0  # ポジティブペアの類似度が入っている位置の値を0に
            )
            mask[batch_size + i, i] = (
                0  # ポジティブペアの類似度が入っている位置の値を0に
            )
        return mask

    def forward(self, z1, z2, _p1, _p2):
        # 全てのサンプル間の類似度の計算
        z = torch.cat((z1, z2), dim=0)  # ネットワークの２つの出力を1つのTensorに
        z = nn.functional.normalize(z, dim=1)  # 正規化
        sim = torch.matmul(z, z.T) / self.T  # 全ての特徴量間のコサイン類似度を計算

        # ポジティブペア/ネガティブペアの類似度の取得
        sim_i_j = torch.diag(
            sim, self.batch_size
        )  # ポジティブペアの類似度(i->j))を抽出
        sim_j_i = torch.diag(
            sim, -self.batch_size
        )  # ポジティブペアの類似度(j->i))を抽出
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(
            self.N, 1
        )  # ポジティブペアの類似度を1つのTensorに
        negative_samples = sim[self.mask].reshape(
            self.N, -1
        )  # ネガティブペアの類似度のみのTensorを作成
        logits = torch.cat(
            (positive_samples, negative_samples), dim=1
        )  # ポジティブペアとネガティブペアの類似度を1つのテンソルに

        # NT-Xent損失の計算
        labels = (
            torch.zeros(self.N).to(positive_samples.cuda()).long()
        )  # ポジティブペアの類似度の位置を表すTensorを作成
        loss = (
            self.criterion(logits, labels) / self.N
        )  # 損失計算（総和）+データ数で除算（平均）
        return loss


class MoCoLoss(nn.Module):
    # https://github.com/facebookresearch/moco/blob/main/moco/builder.py#L154
    def __init__(self, projector, K=4096, T=0.2):
        super(MoCoLoss, self).__init__()
        # InfoNCE損失
        self.K = K
        self.T = T
        self.criterion = nn.CrossEntropyLoss()
        # queue (過去の出力を保存する領域)
        self.register_buffer("queue", torch.randn(projector.out_dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr

    def forward(self, z1, z2, _p1, _p2):
        # 特徴量の正規化
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)

        # compute logits
        l_pos = torch.einsum("nc,nc->n", [z1, z2]).unsqueeze(
            -1
        )  # ポジティブペアとのコサイン類似度
        l_neg = torch.einsum(
            "nc,ck->nk", [z1, self.queue.clone().detach()]
        )  # ネガティブペアとのコサイン類似度
        logits = torch.cat([l_pos, l_neg], dim=1)

        # InfoNCE損失の計算
        logits /= self.T  # 温度パラメータの適用
        labels = torch.zeros(
            logits.shape[0], dtype=torch.long
        ).cuda()  # labels: ポジティブペアの類似度の位置
        self._dequeue_and_enqueue(z2)  # queueの更新
        loss = self.criterion(logits, labels)  # 損失を計算
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


class BYOLLoss(nn.Module):
    # BYOLの論文内の擬似コードを参考に作成
    def __init__(self):
        super(BYOLLoss, self).__init__()
        # コサイン類似度
        self.criterion = nn.CosineSimilarity(dim=1)

    def forward(self, z1, z2, p1, p2):
        # 負のコサイン類似度の計算
        loss = -2 * (self.criterion(p1, z2).mean() + self.criterion(p2, z1).mean())
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

    def off_diagonal(self, x):
        # https://github.com/facebookresearch/barlowtwins/blob/main/main.py#L180
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2, _p1, _p2):
        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)  # バッチ正規化->内積->コサイン類似度
        c.div_(self.batch_size)

        # 損失を計算
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss


class DINOLoss(nn.Module):
    # https://github.com/facebookresearch/barlowtwins/blob/main/main.py#L211
    def __init__(
        self,
        last_out_dim,
        student_T=0.1,
        center_momentum=0.9,
        nmb_crops=8,
        warmup_teacher_T=0.04,
        teacher_T=0.07,
        warmup_teacher_T_epochs=50,
        nepochs=800,
        iteration_per_epoch=800,
    ):
        super(DINOLoss, self).__init__()
        # 損失関数
        self.student_T = student_T
        self.center_momentum = center_momentum
        self.nmb_crops = nmb_crops
        self.register_buffer("center", torch.zeros(1, last_out_dim))
        self.teacher_T_schedule = np.concatenate(
            (
                np.linspace(warmup_teacher_T, teacher_T, warmup_teacher_T_epochs),
                np.ones(nepochs - warmup_teacher_T_epochs) * teacher_T,
            )
        )
        self.iteration_per_epoch = iteration_per_epoch
        self.iteration = 0
        self.epoch = 0

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / (len(teacher_output))
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )  # ema update

    def forward(self, _z1, _z2, student_output, teacher_output):
        self.iteration += 1
        if self.iteration > self.iteration_per_epoch:
            self.iteration = 1
            self.epoch += 1

        student_out = student_output / self.student_T
        student_out = student_out.chunk(self.nmb_crops)  # tensorをViewごとに分割

        # teacher centering and sharpening
        temp = self.teacher_T_schedule[self.epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)  # tensorをViewごとに分割

        t_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue  # we skip cases where student and teacher operate on the same view
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                t_loss += loss.mean()
                n_loss_terms += 1
        t_loss /= n_loss_terms
        self.update_center(teacher_output)
        return t_loss


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, target_output, source_output):
        z1_m1 = target_output[1]
        z2_m1 = target_output[2]
        z1_m2 = source_output[1]
        z2_m2 = source_output[2]

        fvec_m1 = torch.cat((z1_m1, z2_m1), dim=0)
        fvec_m2 = torch.cat((z1_m2, z2_m2), dim=0)

        loss = self.criterion(fvec_m1, fvec_m2.detach())
        return loss


class KLLoss(nn.Module):
    def __init__(self, T=1, lam=1):
        super(KLLoss, self).__init__()
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.criterion = KLDivLoss(T=T)
        self.lam = lam

    def forward(self, target_output, source_output):
        z1_m1 = target_output[1]
        z2_m1 = target_output[2]
        z1_m2 = source_output[1]
        z2_m2 = source_output[2]

        sim_m1 = self.similarity_f(z1_m1.unsqueeze(1), z2_m1.unsqueeze(0))
        sim_m2 = self.similarity_f(z1_m2.unsqueeze(1), z2_m2.unsqueeze(0))

        loss = self.lam * self.criterion(sim_m1, sim_m2)
        return loss


class MSEKLLoss(nn.Module):
    def __init__(self, mse_weight=1.0, kl_weight=1.0, kl_temperature=1.0):
        super(MSEKLLoss, self).__init__()
        self.mse_loss = MSELoss()
        self.kl_loss = KLLoss(T=kl_temperature)

        self.mse_weight = mse_weight
        self.kl_weight = kl_weight

    def forward(self, target_output, source_output):
        mse_loss = self.mse_loss(target_output, source_output)
        kl_loss = self.kl_loss(target_output, source_output)

        loss = self.mse_weight * mse_loss + self.kl_weight * kl_loss
        return loss
