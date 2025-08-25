import torch
import torch.nn as nn
import math


class ThroughGate(nn.Module):
    def __init__(self, max_epoch):
        super(ThroughGate, self).__init__()

    def forward(self, loss, epoch):
        return loss


class CutoffGate(nn.Module):
    def __init__(self, max_epoch):
        super(CutoffGate, self).__init__()

    def forward(self, loss, epoch):
        # 勾配寄与を完全に無効化
        return loss.detach() * 0.0


class PositiveLinearGate(nn.Module):
    def __init__(self, max_epoch):
        super(PositiveLinearGate, self).__init__()
        self.max_epoch = max_epoch

    def forward(self, loss, epoch):
        # 0 -> 1 の線形スケジュール（最終epochで1になるように正規化）
        if self.max_epoch <= 1:
            loss_weight = float(epoch > 0)
        else:
            loss_weight = epoch / (self.max_epoch - 1)
        return loss * loss_weight


class NegativeLinearGate(nn.Module):
    def __init__(self, max_epoch):
        super(NegativeLinearGate, self).__init__()
        self.max_epoch = max_epoch

    def forward(self, loss, epoch):
        # 1 -> 0 の線形スケジュール（最終epochで0になるように正規化）
        if self.max_epoch <= 1:
            loss_weight = float(epoch == 0)
        else:
            loss_weight = (self.max_epoch - 1 - epoch) / (self.max_epoch - 1)
        # 数値誤差を考慮してクリップ
        loss_weight = max(0.0, min(1.0, loss_weight))
        return loss * loss_weight


class SinGate(nn.Module):
    def __init__(self, max_epoch, num_cycles: float = 0.5, phase: float = -math.pi / 2):
        super(SinGate, self).__init__()
        self.max_epoch = max_epoch
        # 固定設定（0〜1 範囲）
        self.min_weight = 0.0
        self.max_weight = 1.0
        # 調整可能なパラメータ
        self.num_cycles = float(num_cycles)
        self.phase = float(phase)

    def forward(self, loss, epoch):
        # sin波形に基づく汎用ゲート。デフォルトは半周期の0→1ランプ。
        if self.max_epoch <= 1:
            t = float(epoch > 0)
        else:
            t = epoch / (self.max_epoch - 1)
            t = max(0.0, min(1.0, t))

        amplitude = (self.max_weight - self.min_weight) / 2.0
        bias = (self.max_weight + self.min_weight) / 2.0
        loss_weight = bias + amplitude * math.sin(
            2.0 * math.pi * self.num_cycles * t + self.phase
        )

        return loss * loss_weight
