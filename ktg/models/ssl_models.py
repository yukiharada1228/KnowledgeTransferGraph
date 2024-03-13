import torch
import torch.nn as nn

from ktg.losses import SimSiamLoss, SwAVLoss


class SimSiamProjector(nn.Module):
    def __init__(self, input_dim, out_dim=2048):
        super(SimSiamProjector, self).__init__()
        self.out_dim = out_dim
        self.projector = nn.Sequential(
            nn.Linear(input_dim, input_dim, bias=False),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, input_dim, bias=False),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim, affine=False),
        )

    def forward(self, x):
        z = self.projector(x)
        return z


class SwAVProjector(nn.Module):
    def __init__(self, input_dim, out_dim=128):
        super(SwAVProjector, self).__init__()
        self.out_dim = out_dim
        self.projector = nn.Sequential(
            nn.Linear(input_dim, 2048),  # Layer1
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, out_dim),
        )  # Layer2

    def forward(self, x):
        z = self.projector(x)
        return z


class SimSiam(nn.Module):
    def __init__(
        self,
        encoder_func,
        batch_size,
        projector_func=SimSiamProjector,
        proj_out_dim=2048,
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
            nn.Linear(proj_out_dim, pred_hidden_dim, bias=False),  # Layer1
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_hidden_dim, proj_out_dim),
        )  # Layer2

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

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        loss = self.criterion(z1.detach(), z2.detach(), p1, p2)
        return [loss, z1, z2]


class SwAV(nn.Module):
    def __init__(
        self,
        encoder_func,
        batch_size,
        projector_func=SwAVProjector,
        proj_out_dim=128,
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
        self.prototypes = nn.Linear(proj_out_dim, nmb_prototypes, bias=False)

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
