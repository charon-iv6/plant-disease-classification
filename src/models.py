import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import logging
from torch import Tensor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResNet9(nn.Module):
    """Custom ResNet9 architecture for plant disease classification
    
    A lightweight ResNet variant with 9 layers, designed for efficient training
    on plant disease images. Features skip connections and batch normalization
    for improved gradient flow and training stability.
    
    Architecture:
        - Initial conv layer (3->64)
        - First residual block (64->128)
        - Second conv block (128->256)
        - Third residual block (256->512)
        - Global pooling and classifier
        
    Attributes:
        conv1: Initial convolution layer
        bn1: Batch normalization for conv1
        conv2: First block main convolution
        bn2: Batch normalization for conv2
        res1: First residual block
        conv3: Second block convolution
        bn3: Batch normalization for conv3
        conv4: Third block main convolution
        bn4: Batch normalization for conv4
        res2: Second residual block
        pool: Max pooling layer
        maxpool: Adaptive average pooling
        dropout: Dropout layer
        classifier: Final classification layers
    """
    
    def __init__(
        self, 
        in_channels: int = 3, 
        num_classes: int = 16, 
        dropout: float = 0.1
    ):
        """Initialize the model
        
        Args:
            in_channels: Number of input channels (default: 3 for RGB)
            num_classes: Number of output classes (default: 16)
            dropout: Dropout rate (default: 0.1)
        """
        super().__init__()
        
        # Initial layer
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # First block
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.res1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128)
        )
        
        # Second block
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Third block
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.res2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512)
        )
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        self.maxpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Initialized ResNet9 with {num_classes} classes")
        
    def _initialize_weights(self) -> None:
        """Initialize model weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Initial conv
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.pool(out)
        
        # First block
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.pool(out)
        out = out + self.res1(out)
        
        # Second block
        out = F.relu(self.bn3(self.conv3(out)))
        out = self.pool(out)
        
        # Third block
        out = F.relu(self.bn4(self.conv4(out)))
        out = self.pool(out)
        out = out + self.res2(out)
        
        # Global pooling and classifier
        out = self.maxpool(out)
        out = out.view(out.size(0), -1)  # Flatten
        out = self.dropout(out)
        out = self.classifier(out)
        
        return out

class ModelEMA:
    """Model Exponential Moving Average
    
    Maintains moving averages of model parameters using exponential weighting.
    This helps improve model robustness and usually achieves better validation accuracy.
    
    Attributes:
        model: The model to maintain moving averages for
        decay: The decay rate for the moving average
        shadow: Dictionary of moving averages
        backup: Backup of original parameters for restoration
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        """Initialize EMA
        
        Args:
            model: Model to maintain moving averages for
            decay: Decay rate for moving average (default: 0.999)
        """
        self.model = model
        self.decay = decay
        self.shadow: Dict[str, Tensor] = {}
        self.backup: Dict[str, Tensor] = {}
        
        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model: nn.Module) -> None:
        """Update moving averages from current model parameters
        
        Args:
            model: Current model to update from
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self) -> None:
        """Apply moving averages to model parameters
        
        Stores backup of current parameters
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self) -> None:
        """Restore original model parameters from backup"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {} 