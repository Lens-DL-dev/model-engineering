import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, embed_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.bn1 = nn.LayerNorm(in_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_dim, embed_dim)
        self.bn2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.normalize(x, dim=1)
        return x

class SimCLRModel(nn.Module):
    def __init__(self, backbone="convnext_tiny", pretrained=True, embed_dim=1024):
        super().__init__()
        if backbone == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            self.encoder = models.convnext_tiny(weights=weights)
            # Remove the classifier
            self.encoder.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),    # Global Average Pooling
                nn.Flatten(),               # Flatten to [batch_size, channels]
                nn.Identity()               # Remove the original linear layer
            )
            in_dim = 768  # ConvNeXt Tiny의 channel 수
            
        elif backbone == "convnext_small":
            weights = models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
            self.encoder = models.convnext_small(weights=weights)
            # Remove the classifier
            self.encoder.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),    # Global Average Pooling
                nn.Flatten(),               # Flatten to [batch_size, channels]
                nn.Identity()               # Remove the original linear layer
            )
            in_dim = 768  # ConvNeXt Small의 channel 수도 768입니다
        else:
            raise NotImplementedError(f"Backbone {backbone} not implemented")
        self.projection_head = ProjectionHead(in_dim, embed_dim)
    
    def forward(self, x):
        features = self.encoder(x)
        embeddings = self.projection_head(features)
        return embeddings
