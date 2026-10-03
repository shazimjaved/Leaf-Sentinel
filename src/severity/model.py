import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class BurdenRegressor(nn.Module):
    def __init__(self, pretrained: bool = True):
        """
        LeafSentinel Phase 4 Lesion Burden Regressor based on EfficientNet-B0.
        
        Outputs a single continuous scalar value for Image-Relative Lesion Burden.
        Uses a raw linear output (no sigmoid).
        
        Args:
            pretrained (bool): If True, load ImageNet pretrained weights.
                Use False when loading a trained checkpoint to avoid
                unnecessary weight downloads.
        """
        super().__init__()
        
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        
        # Replace the final classifier head with a single output node
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input image tensor (B, C, H, W)
            
        Returns:
            torch.Tensor: Raw predicted burden of shape (B, 1)
        """
        return self.backbone(x)
        
    def get_parameter_counts(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params
