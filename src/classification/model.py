import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class DiseaseClassifier(nn.Module):
    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        """
        LeafSentinel Phase 3 Disease Classifier based on EfficientNet-B0.
        
        Args:
            num_classes (int): Number of disease classes to predict.
            pretrained (bool): If True, load ImageNet pretrained weights.
                Use False when loading a trained checkpoint to avoid
                unnecessary weight downloads.
        """
        super().__init__()
        self.num_classes = num_classes
        
        # Use modern torchvision weights API instead of deprecated pretrained=True
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        
        # Replace the final classifier head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input image tensor (B, C, H, W)
            
        Returns:
            torch.Tensor: Raw logits of shape (B, num_classes)
        """
        return self.backbone(x)
        
    def get_parameter_counts(self):
        """Returns total and trainable parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params
