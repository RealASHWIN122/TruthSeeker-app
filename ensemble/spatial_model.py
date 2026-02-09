"""
Spatial Model: RGB texture analysis & blending artifact detection
Focuses on high-frequency inconsistencies typical in GAN/diffusion-generated content
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SpatialModel(nn.Module):
    """
    Independent spatial model for deepfake detection.
    Analyzes RGB texture, lighting inconsistencies, and facial blending artifacts.
    
    Key features:
    - Multi-scale feature extraction
    - Spatial attention for artifact localization
    - High-frequency emphasis via residual connections
    """
    
    def __init__(self, num_classes: int = 2, embedding_dim: int = 512):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        
        # Initial convolution with high-frequency emphasis
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Multi-scale feature extraction blocks
        # Design: Progressive downsampling with increasing channels
        self.layer1 = self._make_layer(64, 128, num_blocks=3, stride=2)
        self.layer2 = self._make_layer(128, 256, num_blocks=4, stride=2)
        self.layer3 = self._make_layer(256, 512, num_blocks=6, stride=2)
        self.layer4 = self._make_layer(512, 512, num_blocks=3, stride=1)
        
        # Spatial attention module
        # Helps focus on regions with artifacts (face boundaries, hair, teeth)
        self.attention = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Global pooling and embedding projection
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Embedding layer
        self.embedding = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
        
        # Reliability estimation head
        # Produces a scalar [0,1] indicating confidence in this model's prediction
        self.reliability_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def _make_layer(self, in_channels: int, out_channels: int, 
                    num_blocks: int, stride: int = 1) -> nn.Module:
        """Create a residual layer with multiple blocks"""
        layers = []
        
        # First block with downsampling
        layers.append(nn.Conv2d(in_channels, out_channels, 
                               kernel_size=3, stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        # Remaining blocks
        for _ in range(num_blocks - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, 
                                   kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial features from frames
        
        Args:
            x: Input tensor [B, T, C, H, W] or [B, C, H, W]
            
        Returns:
            features: [B, 512, H', W']
        """
        # Handle video input by sampling frames
        if len(x.shape) == 5:
            B, T, C, H, W = x.shape
            # Sample up to 8 frames uniformly
            if T > 8:
                indices = torch.linspace(0, T-1, 8, dtype=torch.long, device=x.device)
                x = x[:, indices]
                T = 8
            # Process all frames together
            x = x.reshape(B * T, C, H, W)
        else:
            B = x.size(0)
            T = 1
        
        # Feature extraction pipeline
        x = self.stem(x)           # [B*T, 64, H/4, W/4]
        x = self.layer1(x)         # [B*T, 128, H/8, W/8]
        x = self.layer2(x)         # [B*T, 256, H/16, W/16]
        x = self.layer3(x)         # [B*T, 512, H/32, W/32]
        x = self.layer4(x)         # [B*T, 512, H/32, W/32]
        
        # Apply spatial attention
        attention_map = self.attention(x)  # [B*T, 1, H/32, W/32]
        x = x * attention_map  # Weighted features
        
        if T > 1:
            # Temporal aggregation: average across frames
            spatial_features = x.reshape(B, T, *x.shape[1:])
            x = spatial_features.mean(dim=1)  # [B, 512, H/32, W/32]
        
        return x
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> dict:
        """
        Forward pass with logits, probabilities, and reliability
        
        Args:
            x: Input frames [B, T, C, H, W] or [B, C, H, W]
            return_features: If True, also return intermediate features
            
        Returns:
            dict containing:
                - logits: [B, num_classes] raw classification scores
                - probabilities: [B, num_classes] softmax probabilities
                - reliability: [B] confidence score for this model
                - features: [B, embedding_dim] (if return_features=True)
        """
        # Extract spatial features
        features = self.extract_features(x)  # [B, 512, H', W']
        
        # Global pooling
        pooled = self.global_pool(features).flatten(1)  # [B, 512]
        
        # Embedding
        embedding = self.embedding(pooled)  # [B, embedding_dim]
        
        # Classification
        logits = self.classifier(embedding)  # [B, num_classes]
        probabilities = F.softmax(logits, dim=-1)
        
        # Reliability estimation
        # Higher reliability when features are discriminative
        reliability = self.reliability_head(embedding).squeeze(-1)  # [B]
        
        output = {
            'logits': logits,
            'probabilities': probabilities,
            'reliability': reliability
        }
        
        if return_features:
            output['features'] = embedding
        
        return output


def test_spatial_model():
    """Test the spatial model"""
    model = SpatialModel(num_classes=2, embedding_dim=512)
    
    # Test with single frame
    x_single = torch.randn(4, 3, 224, 224)
    out_single = model(x_single)
    print("Single frame test:")
    print(f"  Logits shape: {out_single['logits'].shape}")
    print(f"  Probs shape: {out_single['probabilities'].shape}")
    print(f"  Reliability shape: {out_single['reliability'].shape}")
    print(f"  Sample probs: {out_single['probabilities'][0]}")
    print(f"  Sample reliability: {out_single['reliability'][0]:.4f}")
    
    # Test with video
    x_video = torch.randn(4, 16, 3, 224, 224)
    out_video = model(x_video, return_features=True)
    print("\nVideo test:")
    print(f"  Logits shape: {out_video['logits'].shape}")
    print(f"  Features shape: {out_video['features'].shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


if __name__ == '__main__':
    test_spatial_model()
