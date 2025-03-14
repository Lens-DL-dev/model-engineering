import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm

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
    
    A100 80GB GPU에 최적화된 버전으로, 더 다양한 백본 옵션을 지원합니다.
    """
    def __init__(self, backbone="convnext_tiny", pretrained=True, embed_dim=1024, use_timm=False):
        super().__init__()
        self.backbone_name = backbone
        self.embed_dim = embed_dim
        
        # timm 라이브러리 사용 여부 (더 다양한 모델에 접근 가능)
        self.use_timm = use_timm
        
        if use_timm:
            # timm 라이브러리 사용 - 더 많은 백본 옵션 지원
            self.encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool='avg')
            if 'convnext' in backbone:
                if 'tiny' in backbone:
                    in_dim = 768
                elif 'small' in backbone:
                    in_dim = 768
                elif 'base' in backbone:
                    in_dim = 1024
                elif 'large' in backbone:
                    in_dim = 1536
                elif 'xlarge' in backbone:
                    in_dim = 2048
                else:
                    raise ValueError(f"Unknown ConvNeXt variant: {backbone}")
            else:
                # 다른 모델의 feature dimension 확인
                in_dim = self.encoder.num_features
        else:
            # torchvision 모델 사용
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
            elif backbone == "convnext_base":
                weights = models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
                self.encoder = models.convnext_base(weights=weights)
                self.encoder.classifier = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Identity()
                )
                in_dim = 1024  # ConvNeXt Base는 1024 차원
            elif backbone == "convnext_large":
                weights = models.ConvNeXt_Large_Weights.IMAGENET1K_V1 if pretrained else None
                self.encoder = models.convnext_large(weights=weights)
                self.encoder.classifier = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Identity()
                )
                in_dim = 1536  # ConvNeXt Large는 1536 차원
            else:
                raise NotImplementedError(f"Backbone {backbone} not implemented for SupConModel")
        
        self.projection_head = ProjectionHead(in_dim, embed_dim)
        
        # A100 GPU에서 성능 최적화를 위한 메모리 캐싱 방지
        self.use_checkpoint = False
        if hasattr(torch.utils.checkpoint, 'checkpoint'):
            self.use_checkpoint = True
    
    def forward(self, x):
        """
        입력 x: [B, C, H, W]
        반환: [B, embed_dim] (정규화된 임베딩)
        """
        if self.use_checkpoint and self.training:
            # 대규모 모델 학습 시 메모리 효율성을 위한 그라디언트 체크포인팅
            from torch.utils.checkpoint import checkpoint
            features = checkpoint(self.encoder, x)
        else:
            features = self.encoder(x)          # [B, in_dim]
        
        embeddings = self.projection_head(features)  # [B, embed_dim]
        return embeddings
    
    def freeze_backbone(self, freeze=True):
        """백본 모델을 고정하고 projection head만 학습하는 옵션"""
        for param in self.encoder.parameters():
            param.requires_grad = not freeze
            
    def freeze_layers_below(self, layer_idx):
        """지정된 레이어 아래의 모든 레이어를 고정하는 옵션 (미세 조정용)"""
        if 'convnext' in self.backbone_name:
            # ConvNeXt 모델의 경우 stages를 기준으로 고정
            if not self.use_timm:
                num_stages = len(self.encoder.features)
                for i, stage in enumerate(self.encoder.features):
                    if i < layer_idx:
                        for param in stage.parameters():
                            param.requires_grad = False
                    else:
                        for param in stage.parameters():
                            param.requires_grad = True
            else:
                # timm 모델의 경우 다른 접근 방식 필요
                # 각 모델마다 구조가 다르므로 일반적인 구현은 어려움
                pass
    
    def get_feature_extractor(self):
        """임베딩 추출용 함수를 반환 (모델 배포 시 활용)"""
        def extract_fn(images):
            self.eval()
            with torch.no_grad():
                return self.forward(images)
        return extract_fn
