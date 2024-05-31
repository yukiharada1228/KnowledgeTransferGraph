import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from bitnetb158.replace_hf import replace_layers_with_bitb158_layers
from torch.nn import init


# ResNet for CIFAR
class DownsampleA(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleA, self).__init__()
        assert stride == 2
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.avg(x)
        return torch.cat((x, x.mul(0)), 1)


class DownsampleC(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleC, self).__init__()
        assert stride != 1 or nIn != nOut
        self.conv = nn.Conv2d(
            nIn, nOut, kernel_size=1, stride=stride, padding=0, bias=False
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class DownsampleD(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleD, self).__init__()
        assert stride == 2
        self.conv = nn.Conv2d(
            nIn, nOut, kernel_size=2, stride=stride, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(nOut)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ResNetBasicblock(nn.Module):
    expansion = 1
    """
    RexNet basicblock (https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua)
    """

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResNetBasicblock, self).__init__()

        self.conv_a = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn_a = nn.BatchNorm2d(planes)

        self.conv_b = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn_b = nn.BatchNorm2d(planes)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        basicblock = self.conv_a(x)
        basicblock = self.bn_a(basicblock)
        basicblock = F.relu(basicblock, inplace=True)

        basicblock = self.conv_b(basicblock)
        basicblock = self.bn_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(x)

        return F.relu(residual + basicblock, inplace=True)


class CifarResNet(nn.Module):
    """
    ResNet optimized for the Cifar dataset, as specified in
    https://arxiv.org/abs/1512.03385.pdf
    """

    def __init__(self, block, depth, num_classes):
        """Constructor
        Args:
            depth: number of layers.
            num_classes: number of classes
            base_width: base width
        """
        super(CifarResNet, self).__init__()

        # Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        assert (depth - 2) % 6 == 0, "depth should be one of 20, 32, 44, 56, 110"
        layer_blocks = (depth - 2) // 6
        # print ('CifarResNet : Depth : {} , Layers for each block : {}'.format(depth, layer_blocks))

        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.inplanes = 16
        self.layer1 = self._make_layer(block, 16, layer_blocks, stride=1)
        self.layer2 = self._make_layer(block, 32, layer_blocks, stride=2)
        self.layer3 = self._make_layer(block, 64, layer_blocks, stride=2)
        self.relu = nn.ReLU(inplace=True)
        # self.avgpool = nn.AvgPool2d(8)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet20(num_classes=10):
    """Constructs a ResNet-20 model for CIFAR-10 (by default)
    Args:
        num_classes (uint): number of classes
    """
    model = CifarResNet(ResNetBasicblock, 20, num_classes)
    return model


def bit_resnet20_b158(num_classes=10):
    model = CifarResNet(ResNetBasicblock, 20, num_classes)
    replace_layers_with_bitb158_layers(model)
    return model


def resnet32(num_classes=10):
    """Constructs a ResNet-32 model for CIFAR-10 (by default)
    Args:
        num_classes (uint): number of classes
    """
    model = CifarResNet(ResNetBasicblock, 32, num_classes)
    return model


def bit_resnet32_b158(num_classes=10):
    model = CifarResNet(ResNetBasicblock, 32, num_classes)
    replace_layers_with_bitb158_layers(model)
    return model


def resnet44(num_classes=10):
    """Constructs a ResNet-44 model for CIFAR-10 (by default)
    Args:
        num_classes (uint): number of classes
    """
    model = CifarResNet(ResNetBasicblock, 44, num_classes)
    return model


def bit_resnet44_b158(num_classes=10):
    model = CifarResNet(ResNetBasicblock, 44, num_classes)
    replace_layers_with_bitb158_layers(model)
    return model


def resnet56(num_classes=10):
    """Constructs a ResNet-56 model for CIFAR-10 (by default)
    Args:
        num_classes (uint): number of classes
    """
    model = CifarResNet(ResNetBasicblock, 56, num_classes)
    return model


def bit_resnet56_b158(num_classes=10):
    model = CifarResNet(ResNetBasicblock, 56, num_classes)
    replace_layers_with_bitb158_layers(model)
    return model


def resnet110(num_classes=10):
    """Constructs a ResNet-110 model for CIFAR-10 (by default)
    Args:
        num_classes (uint): number of classes
    """
    model = CifarResNet(ResNetBasicblock, 110, num_classes)
    return model


def bit_resnet110_b158(num_classes=10):
    model = CifarResNet(ResNetBasicblock, 110, num_classes)
    replace_layers_with_bitb158_layers(model)
    return model


def resnet1202(num_classes=10):
    """Constructs a ResNet-1202 model for CIFAR-10 (by default)
    Args:
        num_classes (uint): number of classes
    """
    model = CifarResNet(ResNetBasicblock, 1202, num_classes)
    return model


def bit_resnet1202_b158(num_classes=10):
    model = CifarResNet(ResNetBasicblock, 1202, num_classes)
    replace_layers_with_bitb158_layers(model)
    return model


def resnet18(num_classes=10):
    model = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    )
    model.maxpool = nn.Identity()
    return model


def bit_resnet18_b158(num_classes=10):
    model = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    )
    model.maxpool = nn.Identity()
    replace_layers_with_bitb158_layers(model)
    return model


def resnet34(num_classes=10):
    model = torchvision.models.resnet34(pretrained=False, num_classes=num_classes)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    )
    model.maxpool = nn.Identity()
    return model

def bit_resnet34_b158(num_classes=10):
    model = torchvision.models.resnet34(pretrained=False, num_classes=num_classes)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    )
    model.maxpool = nn.Identity()
    replace_layers_with_bitb158_layers(model)
    return model


def resnet50(num_classes=10):
    model = torchvision.models.resnet50(pretrained=False, num_classes=num_classes)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    )
    model.maxpool = nn.Identity()
    return model

def bit_resnet50_b158(num_classes=10):
    model = torchvision.models.resnet50(pretrained=False, num_classes=num_classes)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    )
    model.maxpool = nn.Identity()
    replace_layers_with_bitb158_layers(model)
    return model