import copy
import math

import torch
import torch.nn as nn

from ktg.losses import (
    BarlowTwinsLoss,
    BYOLLoss,
    DINOLoss,
    MoCoLoss,
    SimCLRLoss,
    SimSiamLoss,
    SwAVLoss,
)
from ktg.models.projector import (
    BarlowTwinsProjector,
    BYOLProjector,
    DINOProjector,
    MoCoProjector,
    SimCLRProjector,
    SimSiamProjector,
    SwAVProjector,
)


class SimCLR(nn.Module):
    def __init__(
        self,
        encoder_func,
        out_dim=128,
    ):
        super(SimCLR, self).__init__()

        # ネットワークの用意
        self.encoder = encoder_func()
        self.input_dim = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()

        # projector(MLP)の用意
        self.projector = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.input_dim, out_dim, bias=False),
        )

    @torch.no_grad()
    def encoder_features(self, x):
        return self.encoder(x)

    @torch.no_grad()
    def projector_features(self, x):
        return self.projector(self.encoder(x))

    def forward(self, x1, x2, _):  # x1:torch.Tensor, x2:torch.Tensor, x3:list
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)

        z1 = self.projector(h1)
        z2 = self.projector(h2)

        return [z1, z2]


class MoCo(nn.Module):
    # MoCO
    def __init__(self, encoder_func, batch_size, projector_func=MoCoProjector, m=0.999):
        super(MoCo, self).__init__()

        # ネットワークの用意
        self.encoder = encoder_func()
        self.input_dim = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()

        # projector(MLP)の用意
        self.projector = projector_func(input_dim=self.input_dim)

        # 自己教師あり学習の損失
        self.criterion = MoCoLoss(self.projector)

        # momentum encoderの用意
        self.m = m
        self.m_encoder = copy.deepcopy(self.encoder)
        self.m_projector = copy.deepcopy(self.projector)

        # momentum encoderのパラメータがbackpropで更新されないように設定
        for m_param, m_proj_param in zip(
            self.m_encoder.parameters(), self.m_projector.parameters()
        ):
            m_param.requires_grad = False
            m_proj_param.requires_grad = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        # encoder
        for param, m_param in zip(
            self.encoder.parameters(), self.m_encoder.parameters()
        ):
            m_param.data = m_param.data * self.m + param.data * (1.0 - self.m)
        # projector
        for proj_param, m_proj_param in zip(
            self.projector.parameters(), self.m_projector.parameters()
        ):
            m_proj_param.data = m_proj_param.data * self.m + proj_param.data * (
                1.0 - self.m
            )

    @torch.no_grad()
    def encoder_features(self, x):
        return self.encoder(x)

    @torch.no_grad()
    def projector_features(self, x):
        return self.projector(self.encoder(x))

    def forward(self, x1, x2, _):  # x1:torch.Tensor, x2:torch.Tensor, x3:list
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)

        z1 = self.projector(h1)
        z2 = self.projector(h2)

        with torch.no_grad():
            self._momentum_update_key_encoder()  # momentum encoderのパラメータをアップデート
            # ***** 本当はここにShuffling BNの処理を追加 *****
            # Shuffling BN
            # *******************************************
            m_h2 = self.m_encoder(x2)
            m_z2 = self.m_projector(m_h2)

        loss = self.criterion(z1, m_z2, None, None)
        return [loss, z1, z2]


class SimSiam(nn.Module):
    def __init__(
        self,
        encoder_func,
        batch_size,
        projector_func=SimSiamProjector,
        pred_hidden_dim=512,
    ):
        super(SimSiam, self).__init__()
        # 自己教師あり学習の損失
        self.criterion = SimSiamLoss()
        # ネットワークの用意
        self.encoder = encoder_func()
        self.input_dim = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()

        # projector(MLP)の用意
        self.projector = projector_func(input_dim=self.input_dim)

        # predictor(MLP)の用意
        self.predictor = nn.Sequential(
            nn.Linear(self.projector.out_dim, pred_hidden_dim, bias=False),  # Layer1
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_hidden_dim, self.projector.out_dim),
        )  # Layer2

    @torch.no_grad()
    def encoder_features(self, x):
        return self.encoder(x)

    @torch.no_grad()
    def projector_features(self, x):
        return self.projector(self.encoder(x))

    def forward(self, x1, x2, _):  # x1:torch.Tensor, x2:torch.Tensor, x3:None
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)

        z1 = self.projector(h1)
        z2 = self.projector(h2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        loss = self.criterion(z1.detach(), z2.detach(), p1, p2)
        return [loss, z1, z2]


class BYOL(nn.Module):
    def __init__(
        self,
        encoder_func,
        batch_size,
        projector_func=BYOLProjector,
        pred_hidden_dim=4096,
        base_m=0.996,
        max_iteration=1,
    ):
        super(BYOL, self).__init__()

        # 自己教師あり学習の損失
        self.criterion = BYOLLoss()

        # ネットワークの用意
        self.encoder = encoder_func()
        self.input_dim = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()

        # projector(MLP)の用意
        self.projector = projector_func(input_dim=self.input_dim)

        # predictor(MLP)の用意
        # https://github.com/deepmind/deepmind-research/blob/master/byol/utils/networks.py#L40
        self.predictor = nn.Sequential(
            nn.Linear(self.projector.out_dim, pred_hidden_dim, bias=True),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_hidden_dim, self.projector.out_dim, bias=False),
        )

        # momentum周りの設定
        self.m = 1.0
        self.base_m = base_m
        self.max_iteration = max_iteration
        self.iteration = 0

        # momentum encoderの用意
        self.m_encoder = copy.deepcopy(self.encoder)
        self.m_projector = copy.deepcopy(self.projector)

        # momentum encoderのパラメータがbackpropで更新されないように設定
        for m_param, m_proj_param in zip(
            self.m_encoder.parameters(), self.m_projector.parameters()
        ):
            m_param.requires_grad = False
            m_proj_param.requires_grad = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        # encoder
        for param, m_param in zip(
            self.encoder.parameters(), self.m_encoder.parameters()
        ):
            m_param.data = m_param.data * self.m + param.data * (1.0 - self.m)
        # projector
        for proj_param, m_proj_param in zip(
            self.projector.parameters(), self.m_projector.parameters()
        ):
            m_proj_param.data = m_proj_param.data * self.m + proj_param.data * (
                1.0 - self.m
            )

    @torch.no_grad()
    def encoder_features(self, x):
        return self.encoder(x)

    @torch.no_grad()
    def projector_features(self, x):
        return self.projector(self.encoder(x))

    def forward(self, x1, x2, x3):  # x1:torch.Tensor, x2:torch.Tensor, x3:list
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)

        z1 = self.projector(h1)
        z2 = self.projector(h2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        with torch.no_grad():
            self.iteration += 1
            self.m = 1.0 - (1.0 - self.base_m) * 0.5 * (
                1.0 + math.cos(math.pi * (self.iteration) / (self.max_iteration))
            )  # momentumの値を更新(論文記載の式： 1−(1−τ_base)·(cos(πk/K)+1)/2 )
            self._momentum_update_key_encoder()  # momentum encoderのパラメータをアップデート
            m_h1 = self.m_encoder(x1)
            m_h2 = self.m_encoder(x2)

            m_z1 = self.m_projector(m_h1)
            m_z2 = self.m_projector(m_h2)

        loss = self.criterion(m_z1, m_z2, p1, p2)
        return [loss, z1, z2]


class BarlowTwins(nn.Module):
    def __init__(self, encoder_func, batch_size, projector_func=BarlowTwinsProjector):
        super(BarlowTwins, self).__init__()

        # ネットワークの用意
        self.encoder = encoder_func()
        self.input_dim = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()

        # projector(MLP)の用意
        self.projector = projector_func(input_dim=self.input_dim)

        # 自己教師あり学習の損失
        self.criterion = BarlowTwinsLoss(batch_size, self.projector)

    @torch.no_grad()
    def encoder_features(self, x):
        return self.encoder(x)

    @torch.no_grad()
    def projector_features(self, x):
        return self.projector(self.encoder(x))

    def forward(self, x1, x2, _):  # x1:torch.Tensor, x2:torch.Tensor, x3:None
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)

        z1 = self.projector(h1)
        z2 = self.projector(h2)

        loss = self.criterion(z1, z2, None, None)
        return [loss, z1, z2]


class SwAV(nn.Module):
    def __init__(
        self,
        encoder_func,
        batch_size,
        projector_func=SwAVProjector,
        nmb_prototypes=3000,
    ):
        super(SwAV, self).__init__()
        # 自己教師あり学習の損失
        self.criterion = SwAVLoss(batch_size)

        # ネットワークの用意
        self.encoder = encoder_func()
        self.input_dim = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()

        # projector(MLP)の用意
        self.projector = projector_func(input_dim=self.input_dim)

        # クラスタの確率分布を出力するlayerの用意（nmb_prototypes：クラスタ数）
        # -> つまり，この層の重みパラメータ＝プロトタイプベクトル
        self.prototypes = nn.Linear(self.projector.out_dim, nmb_prototypes, bias=False)

    @torch.no_grad()
    def encoder_features(self, x):
        return self.encoder(x)

    @torch.no_grad()
    def projector_features(self, x):
        return self.projector(self.encoder(x))

    def forward(self, x1, x2, x3):  # x1:torch.Tensor, x2:torch.Tensor, x3:list
        # ミニバッチサイズの取得
        batch_size = x1.shape[0]

        # 入力画像を1つのリストに
        x1 = [x1, x2] + x3

        # normalize the prototypes
        with torch.no_grad():
            w = self.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            self.prototypes.weight.copy_(w)

        # x1: マルチクロップした画像が格納されたリスト [x_1,x_2,x_3,x_4,...], x_1:ミニバッチ数分のデータ
        # idx_cropsの例 : tensor([2, 8])
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x1]), return_counts=True
            )[1],
            0,
        )

        start_idx = 0
        for end_idx in idx_crops:
            h1 = self.encoder(torch.cat(x1[start_idx:end_idx]).cuda(non_blocking=True))
            if start_idx == 0:
                output = h1
            else:
                output = torch.cat((output, h1))
            start_idx = end_idx

        z1 = self.projector(output)
        p1 = self.prototypes(
            nn.functional.normalize(z1, dim=1, p=2)
        )  # 各クラスタのクラススコア(logits=コサイン類似度)

        loss = self.criterion(
            nn.functional.normalize(z1, dim=1, p=2).detach(), None, p1, None
        )
        return [loss, z1[0:batch_size], z1[batch_size : batch_size * 2]]


class DINO(nn.Module):
    def __init__(
        self,
        encoder_func,
        batch_size,
        projector_func=DINOProjector,
        last_out_dim=60000,
        base_m=0.996,
        max_iteration=1,
    ):
        super(DINO, self).__init__()

        # 自己教師あり学習の損失
        self.criterion = DINOLoss(last_out_dim)

        # ネットワークの用意
        self.encoder = encoder_func()
        self.input_dim = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()

        # projector(MLP)の用意
        self.projector = projector_func(input_dim=self.input_dim)

        # クラススコアを出力する層
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(self.projector.out_dim, last_out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False

        # momentum周りの設定
        self.m = 1.0
        self.base_m = base_m
        self.max_iteration = max_iteration
        self.iteration = 0

        # momentum encoderの用意
        self.m_encoder = copy.deepcopy(self.encoder)
        self.m_projector = copy.deepcopy(self.projector)

        self.m_last_layer = nn.utils.weight_norm(
            nn.Linear(self.projector.out_dim, last_out_dim, bias=False)
        )  # nn.utils.weight_norm:forwardが呼び出されるたびにm_last_layerの重みパラメータを正規化
        self.m_last_layer.weight_g.data.fill_(
            1
        )  # 正規化後の重みの値をシフトさせるパラメータgの初期値を１に設定
        self.m_last_layer.weight_g.requires_grad = False  # 正規化後の重みの値をシフトさせるパラメータgをパラメータ更新の対象から外す
        self.m_last_layer.load_state_dict(
            self.last_layer.state_dict()
        )  # self.last_layerをdeepcopyしようとしたらエラーが出たので代替案（DINOで使用されているパラメータを一致させる方法）

        # momentum encoderのパラメータがbackpropで更新されないように設定
        for m_param, m_proj_param, m_last_param in zip(
            self.m_encoder.parameters(),
            self.m_projector.parameters(),
            self.m_last_layer.parameters(),
        ):
            m_param.requires_grad = False
            m_proj_param.requires_grad = False
            m_last_param.requires_grad = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        # encoder
        for param, m_param in zip(
            self.encoder.parameters(), self.m_encoder.parameters()
        ):
            m_param.data = m_param.data * self.m + param.data * (1.0 - self.m)
        # projector
        for proj_param, m_proj_param in zip(
            self.projector.parameters(), self.m_projector.parameters()
        ):
            m_proj_param.data = m_proj_param.data * self.m + proj_param.data * (
                1.0 - self.m
            )
        # last layer
        for last_param, m_last_param in zip(
            self.last_layer.parameters(), self.m_last_layer.parameters()
        ):
            m_last_param.data = m_last_param.data * self.m + last_param.data * (
                1.0 - self.m
            )

    @torch.no_grad()
    def encoder_features(self, x):
        return self.encoder(x)

    @torch.no_grad()
    def projector_features(self, x):
        z = self.projector(self.encoder(x))
        z = nn.functional.normalize(z, dim=-1, p=2)
        p = self.last_layer(z)
        return p

    def forward(self, x1, x2, x3):  # x1:torch.Tensor, x2:torch.Tensor, x3:list
        # ミニバッチサイズの取得
        batch_size = x1.shape[0]

        # 入力画像を1つのリストに
        x1 = [x1, x2] + x3

        # x1: マルチクロップした画像が格納されたリスト [x_1,x_2,x_3,x_4,...], x_1:ミニバッチ数分のデータ
        # idx_cropsの例 : tensor([2, 8])
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x1]), return_counts=True
            )[1],
            0,
        )

        # encoderの処理
        start_idx = 0
        for end_idx in idx_crops:
            h1 = self.encoder(torch.cat(x1[start_idx:end_idx]).cuda(non_blocking=True))
            if start_idx == 0:
                output = h1
            else:
                output = torch.cat((output, h1))
            start_idx = end_idx

        z1 = self.projector(output)
        p1 = self.last_layer(nn.functional.normalize(z1, dim=-1, p=2))

        # momentum encoderの処理
        with torch.no_grad():
            self.iteration += 1
            self.m = 1.0 - (1.0 - self.base_m) * 0.5 * (
                1.0 + math.cos(math.pi * (self.iteration) / (self.max_iteration))
            )  # momentumの値を更新(論文記載の式： 1−(1−τ_base)·(cos(πk/K)+1)/2 )
            self._momentum_update_key_encoder()  # momentum encoderのパラメータをアップデート

            # global cropの画像のみ入力
            start_idx = 0
            end_idx = idx_crops[0]
            m_h1 = self.m_encoder(
                torch.cat(x1[start_idx:end_idx]).cuda(non_blocking=True)
            )
            m_z1 = self.m_projector(m_h1)
            m_z1 = nn.functional.normalize(m_z1, dim=-1, p=2)
            m_p1 = self.m_last_layer(m_z1)

        loss = self.criterion(None, None, p1, m_p1)
        return [loss, z1[0:batch_size], z1[batch_size : batch_size * 2]]
