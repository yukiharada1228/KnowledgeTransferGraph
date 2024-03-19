import torch.nn as nn


class SimCLRProjector(nn.Module):
    def __init__(self, input_dim, out_dim=128):
        super(SimCLRProjector, self).__init__()
        self.out_dim = out_dim
        self.projector = nn.Sequential(
            nn.Linear(input_dim, input_dim, bias=True),
            nn.ReLU(),
            nn.Linear(input_dim, out_dim, bias=False),
        )

    def forward(self, x):
        z = self.projector(x)
        return z


class MoCoProjector(nn.Module):
    def __init__(self, input_dim, out_dim=128):
        super(MoCoProjector, self).__init__()
        self.out_dim = out_dim
        self.projector = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, out_dim),
        )

    def forward(self, x):
        z = self.projector(x)
        return z


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


class BYOLProjector(nn.Module):
    def __init__(self, input_dim, out_dim=256):
        super(BYOLProjector, self).__init__()
        self.out_dim = out_dim
        self.projector = nn.Sequential(
            nn.Linear(input_dim, 4096, bias=True),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, out_dim, bias=False),
        )

    def forward(self, x):
        z = self.projector(x)
        return z


class BarlowTwinsProjector(nn.Module):
    def __init__(self, input_dim, out_dim=8192):
        super(BarlowTwinsProjector, self).__init__()
        self.out_dim = out_dim
        self.projector = nn.Sequential(
            nn.Linear(input_dim, 8192, bias=False),
            nn.BatchNorm1d(8192),
            nn.ReLU(inplace=True),
            nn.Linear(8192, 8192, bias=False),
            nn.BatchNorm1d(8192, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(8192, out_dim, bias=False),
        )

    def forward(self, x):
        z = self.projector(x)
        return z


class SwAVProjector(nn.Module):
    def __init__(self, input_dim, out_dim=128):
        super(SwAVProjector, self).__init__()
        self.out_dim = out_dim
        self.projector = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, out_dim),
        )

    def forward(self, x):
        z = self.projector(x)
        return z


class DINOProjector(nn.Module):
    def __init__(self, input_dim, out_dim=256):
        # https://github.com/facebookresearch/dino/blob/cb711401860da580817918b9167ed73e3eef3dcf/vision_transformer.py#L257
        super(DINOProjector, self).__init__()
        self.out_dim = out_dim
        self.projector = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Linear(2048, out_dim),
        )

    def forward(self, x):
        z = self.projector(x)
        return z
