import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

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
        return F.normalize(x, dim=1)

class ConvNextModel(nn.Module):
    """
    SupCon 방식 학습용 모델.
    ConvNeXt 백본(예: convnext_tiny 또는 convnext_small)과 Projection Head를 통해
    입력 이미지로부터 임베딩을 추출합니다.
    학습 시, 두 view (예, 착용 이미지와 상품 이미지)를 개별적으로 인코딩한 후,
    SupConLoss에서 [B, 2, D] 형태로 스택하여 사용합니다.
    """
    def __init__(self, backbone="convnext_tiny", pretrained=True, embed_dim=1024):
        super().__init__()
        if backbone == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            self.encoder = models.convnext_tiny(weights=weights)
            # classifier 제거: Global Average Pooling 후 Flatten
            self.encoder.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Identity()
            )
            in_dim = 768  # ConvNeXt Tiny의 마지막 feature dimension
        elif backbone == "convnext_small":
            weights = models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
            self.encoder = models.convnext_small(weights=weights)
            self.encoder.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Identity()
            )
            in_dim = 768  # ConvNeXt Small도 768 차원
        else:
            raise NotImplementedError(f"Backbone {backbone} not implemented for SupConModel")
        
        self.projection_head = ProjectionHead(in_dim, embed_dim)
    
    def forward(self, x):
        """
        입력 x: [B, C, H, W]
        반환: [B, embed_dim] (정규화된 임베딩)
        """
        features = self.encoder(x)          # [B, in_dim]
        embeddings = self.projection_head(features)  # [B, embed_dim]
        return embeddings
