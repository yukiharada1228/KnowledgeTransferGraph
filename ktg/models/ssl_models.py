import torch
import torch.nn as nn

from ktg.losses import SimSiamLoss


def SimSiam_projector(input_dim):
    # https://github.com/facebookresearch/simsiam/blob/main/simsiam/builder.py#L28
    out_dim = 2048
    projector = nn.Sequential(
        nn.Linear(input_dim, input_dim, bias=False),  # Layer1
        nn.BatchNorm1d(input_dim),
        nn.ReLU(inplace=True),
        nn.Linear(input_dim, input_dim, bias=False),  # Layer2
        nn.BatchNorm1d(input_dim),
        nn.ReLU(inplace=True),
        nn.Linear(input_dim, out_dim, bias=False),  # Layer3
        nn.BatchNorm1d(out_dim, affine=False),
    )
    return projector


class SimSiam(nn.Module):
    def __init__(
        self,
        encoder_func,
        projector_func=SimSiam_projector,
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

    def forward(self, x1, x2, x3):  # x1:torch.Tensor, x2:torch.Tensor, x3:list
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)

        z1 = self.projector(h1)
        z2 = self.projector(h2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        loss = self.criterion(z1.detach(), z2.detach(), p1, p2)
        return [loss, z1, z2]
