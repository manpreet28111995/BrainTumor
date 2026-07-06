"""
Lightweight wrappers for baseline classification models.
Default settings train from scratch (no pretrained weights) for fair comparison
with FLUX-Net. Set pretrained=True only for a separate ImageNet-pretrained table.
"""

import torch
import torch.nn as nn
from typing import Callable, Dict


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class LightCNN(nn.Module):
    """Simple 4-block CNN — reproduces the paper's baseline architecture."""

    def __init__(self, num_classes=4, base_dim=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_dim, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_dim),
            nn.GELU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        dims = [base_dim, base_dim * 2, base_dim * 4, base_dim * 8]
        self.stages = nn.ModuleList()
        self.stages.append(ConvBlock(dims[0], dims[0]))
        self.stages.append(ConvBlock(dims[0], dims[1], stride=2))
        self.stages.append(ConvBlock(dims[1], dims[2], stride=2))
        self.stages.append(ConvBlock(dims[2], dims[3], stride=2))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dims[3], 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return {"logits": self.head(x)}


def _resnet_block(in_c, out_c, stride=1, downsample=None):
    layers = [
        nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.GELU(),
        nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
    ]
    if downsample is not None:
        return nn.Sequential(*layers), downsample
    return nn.Sequential(*layers), nn.Identity()


class ResNet50(nn.Module):
    """ResNet-50 trained from scratch."""

    def __init__(self, num_classes=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        channels = [64, 256, 512, 1024, 2048]
        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_layer(256, 128, 4, stride=2)
        self.layer3 = self._make_layer(512, 256, 6, stride=2)
        self.layer4 = self._make_layer(1024, 512, 3, stride=2)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(2048, num_classes),
        )
        self._init_weights()

    def _make_layer(self, in_c, mid_c, blocks, stride):
        layers = []
        for i in range(blocks):
            s = stride if i == 0 else 1
            c_in = in_c if i == 0 else mid_c * 4
            downsample = None
            if s != 1 or c_in != mid_c * 4:
                downsample = nn.Sequential(
                    nn.Conv2d(c_in, mid_c * 4, 1, stride=s, bias=False),
                    nn.BatchNorm2d(mid_c * 4),
                )
            layers.append(Bottleneck(c_in, mid_c, s, downsample))
            in_c = mid_c * 4
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return {"logits": self.head(x)}


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_c, mid_c, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, mid_c, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_c)
        self.conv2 = nn.Conv2d(mid_c, mid_c, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_c)
        self.conv3 = nn.Conv2d(mid_c, mid_c * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_c * 4)
        self.act = nn.GELU()
        self.downsample = downsample

    def forward(self, x):
        identity = self.downsample(x) if self.downsample is not None else x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += identity
        return self.act(out)


def _efficientnet_b0_config():
    """MBConv configs for EfficientNet-B0: (expand_ratio, channels, repeats, stride, kernel)"""
    return [
        (1, 16, 1, 1, 3),
        (6, 24, 2, 2, 3),
        (6, 40, 2, 2, 5),
        (6, 80, 3, 2, 3),
        (6, 112, 3, 1, 5),
        (6, 192, 4, 2, 5),
        (6, 320, 1, 1, 3),
    ]


class MBConv(nn.Module):
    def __init__(self, in_c, out_c, expand_ratio, stride, kernel, se_ratio=0.25):
        super().__init__()
        hidden = in_c * expand_ratio
        layers = []
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_c, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.GELU(),
            ]
        layers += [
            nn.Conv2d(hidden, hidden, kernel, stride=stride, padding=kernel // 2, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
        ]
        se_dim = max(1, int(in_c * se_ratio))
        layers += [
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, se_dim, 1),
            nn.GELU(),
            nn.Conv2d(se_dim, hidden, 1),
            nn.Sigmoid(),
        ]
        self.se = nn.Sequential(*layers[-4:])
        self.conv = nn.Sequential(*layers[:-4])
        self.project = nn.Conv2d(hidden, out_c, 1, bias=False)
        self.bn_final = nn.BatchNorm2d(out_c)
        self.use_res = stride == 1 and in_c == out_c

    def forward(self, x):
        shortcut = x
        x = self.conv(x)
        se = self.se(x)
        x = x * se
        x = self.bn_final(self.project(x))
        if self.use_res:
            x += shortcut
        return x


class EfficientNetB0(nn.Module):
    """EfficientNet-B0 trained from scratch."""

    def __init__(self, num_classes=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        config = _efficientnet_b0_config()
        in_c = 32
        self.blocks = nn.ModuleList()
        for expand_ratio, out_c, repeats, stride, kernel in config:
            for i in range(repeats):
                s = stride if i == 0 else 1
                self.blocks.append(MBConv(in_c, out_c, expand_ratio, s, kernel))
                in_c = out_c
        self.head = nn.Sequential(
            nn.Conv2d(in_c, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(1280, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return {"logits": self.head(x)}


class TorchvisionClassifier(nn.Module):
    """Wrap a torchvision classifier so Trainer receives {"logits": tensor}."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return {"logits": self.model(x)}

    def count_parameters(self):
        return {
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }


def _torchvision_weights(weights_enum, pretrained: bool):
    if not pretrained:
        return None
    return weights_enum.DEFAULT


def _attach_count_parameters(model: nn.Module):
    def count_parameters():
        return {
            "total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }

    model.count_parameters = count_parameters
    return model


def _replace_linear(module: nn.Module, attr: str, num_classes: int):
    old = getattr(module, attr)
    if not isinstance(old, nn.Linear):
        raise TypeError(f"Expected {attr} to be nn.Linear, got {type(old)}")
    setattr(module, attr, nn.Linear(old.in_features, num_classes))


def _build_tv_resnet50(num_classes: int, pretrained: bool):
    from torchvision.models import ResNet50_Weights, resnet50

    model = resnet50(weights=_torchvision_weights(ResNet50_Weights, pretrained))
    _replace_linear(model, "fc", num_classes)
    return TorchvisionClassifier(model)


def _build_tv_efficientnet_b0(num_classes: int, pretrained: bool):
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    model = efficientnet_b0(
        weights=_torchvision_weights(EfficientNet_B0_Weights, pretrained)
    )
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return TorchvisionClassifier(model)


def _build_mobilenet_v3_large(num_classes: int, pretrained: bool):
    from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

    model = mobilenet_v3_large(
        weights=_torchvision_weights(MobileNet_V3_Large_Weights, pretrained)
    )
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return TorchvisionClassifier(model)


def _build_convnext_tiny(num_classes: int, pretrained: bool):
    from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

    model = convnext_tiny(
        weights=_torchvision_weights(ConvNeXt_Tiny_Weights, pretrained)
    )
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return TorchvisionClassifier(model)


def _build_vit_b_16(num_classes: int, pretrained: bool):
    from torchvision.models import ViT_B_16_Weights, vit_b_16

    model = vit_b_16(weights=_torchvision_weights(ViT_B_16_Weights, pretrained))
    _replace_linear(model.heads, "head", num_classes)
    return TorchvisionClassifier(model)


def _build_swin_tiny(num_classes: int, pretrained: bool):
    from torchvision.models import Swin_T_Weights, swin_t

    model = swin_t(weights=_torchvision_weights(Swin_T_Weights, pretrained))
    _replace_linear(model, "head", num_classes)
    return TorchvisionClassifier(model)


def _build_lightcnn(num_classes: int, pretrained: bool):
    if pretrained:
        raise ValueError("LightCNN has no pretrained weights.")
    return _attach_count_parameters(LightCNN(num_classes=num_classes))


def _build_custom_resnet50(num_classes: int, pretrained: bool):
    if pretrained:
        raise ValueError("Custom ResNet50 has no pretrained weights.")
    return _attach_count_parameters(ResNet50(num_classes=num_classes))


def _build_custom_efficientnet_b0(num_classes: int, pretrained: bool):
    if pretrained:
        raise ValueError("Custom EfficientNetB0 has no pretrained weights.")
    return _attach_count_parameters(EfficientNetB0(num_classes=num_classes))


BASELINE_BUILDERS: Dict[str, Callable[[int, bool], nn.Module]] = {
    "lightcnn": _build_lightcnn,
    "custom_resnet50": _build_custom_resnet50,
    "custom_efficientnet_b0": _build_custom_efficientnet_b0,
    "resnet50": _build_tv_resnet50,
    "efficientnet_b0": _build_tv_efficientnet_b0,
    "mobilenet_v3_large": _build_mobilenet_v3_large,
    "convnext_tiny": _build_convnext_tiny,
    "vit_b_16": _build_vit_b_16,
    "swin_tiny": _build_swin_tiny,
}


ALIASES = {
    "efficientnet": "efficientnet_b0",
    "mobilenetv3": "mobilenet_v3_large",
    "mobilenet_v3": "mobilenet_v3_large",
    "convnext": "convnext_tiny",
    "vit": "vit_b_16",
    "swin": "swin_tiny",
}


def available_baselines():
    return tuple(BASELINE_BUILDERS.keys())


def build_baseline_model(
    name: str,
    num_classes: int = 4,
    pretrained: bool = False,
) -> nn.Module:
    key = name.lower().replace("-", "_")
    key = ALIASES.get(key, key)
    if key not in BASELINE_BUILDERS:
        valid = ", ".join(available_baselines())
        raise ValueError(f"Unknown baseline '{name}'. Available: {valid}")
    return BASELINE_BUILDERS[key](num_classes, pretrained)
