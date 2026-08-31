"""U-Net architecture with pretrained ResNet-18 encoder for LeafSentinel (Phase 2).

Combines ImageNet-pretrained ResNet-18 feature extraction with a multi-scale
convolutional decoder and skip connections to produce pixel-level lesion logits.
"""

import logging
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights

logger = logging.getLogger(__name__)


class DoubleConv(nn.Module):
    """Two consecutive Conv2d -> BatchNorm -> ReLU blocks with optional residual connection."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class DecoderBlock(nn.Module):
    """Upsampling block combining bilinear upsampling, skip-connection concatenation, and DoubleConv."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Upsample x to match skip spatial dimensions
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResNetUNet(nn.Module):
    """U-Net with ResNet-18 encoder for binary lesion segmentation."""

    def __init__(
        self,
        pretrained: bool = True,
        in_channels: int = 3,
        num_classes: int = 1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Load ResNet-18 encoder
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        try:
            base_resnet = models.resnet18(weights=weights)
        except Exception as e:
            logger.warning(f"Could not load online pretrained weights ({e}). Initializing randomly.")
            base_resnet = models.resnet18(weights=None)

        # Handle input channels != 3 if necessary
        if in_channels != 3:
            self.stem_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.stem_conv = base_resnet.conv1

        self.stem_bn = base_resnet.bn1
        self.stem_relu = base_resnet.relu
        self.maxpool = base_resnet.maxpool

        # Encoder stages
        self.encoder1 = base_resnet.layer1  # 64 ch, stride 4
        self.encoder2 = base_resnet.layer2  # 128 ch, stride 8
        self.encoder3 = base_resnet.layer3  # 256 ch, stride 16
        self.encoder4 = base_resnet.layer4  # 512 ch, stride 32 (bottleneck)

        # Decoder stages
        self.dec4 = DecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
        self.dec3 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        self.dec2 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        self.dec1 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=32)

        # Final head to original resolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]

        # Stem (skip 0)
        s0 = self.stem_relu(self.stem_bn(self.stem_conv(x)))  # (B, 64, H/2, W/2)
        p0 = self.maxpool(s0)                                 # (B, 64, H/4, W/4)

        # Encoder stages
        s1 = self.encoder1(p0)  # (B, 64, H/4, W/4)
        s2 = self.encoder2(s1)  # (B, 128, H/8, W/8)
        s3 = self.encoder3(s2)  # (B, 256, H/16, W/16)
        b4 = self.encoder4(s3)  # (B, 512, H/32, W/32) (Bottleneck)

        # Decoder stages
        d4 = self.dec4(b4, s3)  # (B, 256, H/16, W/16)
        d3 = self.dec3(d4, s2)  # (B, 128, H/8, W/8)
        d2 = self.dec2(d3, s1)  # (B, 64, H/4, W/4)
        d1 = self.dec1(d2, s0)  # (B, 32, H/2, W/2)

        # Upsample back to exact input resolution
        d0 = F.interpolate(d1, size=input_size, mode="bilinear", align_corners=False)
        logits = self.final_conv(d0)  # (B, num_classes, H, W)
        return logits

    def count_parameters(self) -> Dict[str, int]:
        """Count total and trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params
        }


def build_segmentation_model(config: Dict[str, Any]) -> ResNetUNet:
    """Instantiate segmentation model from configuration dictionary."""
    model_cfg = config.get("model", {})
    pretrained = bool(model_cfg.get("pretrained", True))
    in_channels = int(model_cfg.get("in_channels", 3))
    num_classes = int(model_cfg.get("num_classes", 1))

    model = ResNetUNet(
        pretrained=pretrained,
        in_channels=in_channels,
        num_classes=num_classes
    )
    param_info = model.count_parameters()
    logger.info(
        f"Built ResNet18-UNet: {param_info['total_parameters']:,} parameters "
        f"({param_info['trainable_parameters']:,} trainable), in_channels={in_channels}, num_classes={num_classes}."
    )
    return model
