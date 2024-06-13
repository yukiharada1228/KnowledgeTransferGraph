import torchvision
from bitnetb158.replace_hf import replace_layers_with_bitb158_layers


def resnet152(num_classes=10):
    model = torchvision.models.resnet152(pretrained=False, num_classes=num_classes)
    return model


def bit_resnet152_b158(num_classes=10):
    model = torchvision.models.resnet152(pretrained=False, num_classes=num_classes)
    replace_layers_with_bitb158_layers(model)
    return model
