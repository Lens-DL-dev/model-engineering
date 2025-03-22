import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
from transformers import AutoImageProcessor, AutoModel  # 허깅페이스 transformers 라이브러리 추가

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
    def __init__(self, backbone="convnext_tiny", pretrained=True, embed_dim=1024, use_timm=False, use_hf=False):
        super().__init__()
        self.backbone_name = backbone
        self.embed_dim = embed_dim
        
        # 허깅페이스 모델 사용 여부
        self.use_hf = use_hf
        # timm 라이브러리 사용 여부 (더 다양한 모델에 접근 가능)
        self.use_timm = use_timm and not use_hf  # 허깅페이스 모델 사용 시 timm은 무시
        
        if use_hf:
            # 허깅페이스 transformers 라이브러리 사용
            self.encoder = AutoModel.from_pretrained(backbone)
            
            # ConvNextV2 모델의 경우 출력 특성 차원 설정
            if 'convnextv2-base' in backbone:
                in_dim = 1024
            elif 'convnextv2-large' in backbone:
                in_dim = 1536
            elif 'convnextv2-huge' in backbone:
                in_dim = 2048
            elif 'convnextv2-tiny' in backbone:
                in_dim = 768
            elif 'convnextv2-small' in backbone:
                in_dim = 768
            else:
                # 기타 허깅페이스 모델의 경우
                # 설정 파일에서 차원 정보 가져오기
                try:
                    in_dim = self.encoder.config.hidden_sizes[-1]
                except:
                    in_dim = self.encoder.config.hidden_size
            
            # 허깅페이스 모델 출력을 처리하기 위한 풀링 및 플래튼 레이어 추가
            self.pooling = nn.AdaptiveAvgPool2d(1)
            self.flatten = nn.Flatten()
        
        elif use_timm:
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
            if self.use_hf:
                # Hugging Face 모델 체크포인팅
                features = checkpoint(self._forward_features, x)
            else:
                features = checkpoint(self.encoder, x)
        else:
            if self.use_hf:
                # Hugging Face 모델 forward
                features = self._forward_features(x)
            else:
                features = self.encoder(x)  # [B, in_dim]
        
        embeddings = self.projection_head(features)  # [B, embed_dim]
        return embeddings
    
    def _forward_features(self, x):
        """허깅페이스 모델의 특성 추출 처리"""
        if self.use_hf:
            # 허깅페이스 ConvNeXt 모델은 last_hidden_state가 [B, C, H, W] 형태로 출력
            outputs = self.encoder(pixel_values=x)
            
            # last_hidden_state는 [B, C, H, W] 형태
            features = outputs.last_hidden_state
            
            # torchvision 모델과 유사하게 처리: Global Average Pooling 후 Flatten
            features = self.pooling(features)
            features = self.flatten(features)
            
            return features
        else:
            return self.encoder(x)
    
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

class TwoTowerModel(nn.Module):
    """
    Two-Tower 구조:
      - tower_wear: 착용 이미지를 위한 인코더
      - tower_prod: 상품 이미지를 위한 인코더
    
    각 타워는 독립적인 ConvNextModel로 구성되어 있으며, 
    입력 이미지 도메인에 특화된 임베딩을 학습할 수 있습니다.
    """
    def __init__(self, backbone_wear="convnext_tiny", backbone_prod="convnext_tiny",
                 pretrained=True, embed_dim=512, use_timm=False, use_hf=False):
        super().__init__()
        self.tower_wear = ConvNextModel(backbone=backbone_wear, pretrained=pretrained, 
                                        embed_dim=embed_dim, use_timm=use_timm, use_hf=use_hf)
        self.tower_prod = ConvNextModel(backbone=backbone_prod, pretrained=pretrained, 
                                        embed_dim=embed_dim, use_timm=use_timm, use_hf=use_hf)

    def forward(self, wear_img=None, prod_img=None):
        """
        착용 이미지와 상품 이미지 모두 혹은 하나만 입력 가능
        
        Args:
            wear_img: 착용 이미지 텐서, shape [B, C, H, W]
            prod_img: 상품 이미지 텐서, shape [B, C, H, W]
            
        Returns:
            emb_wear: 착용 이미지 임베딩 (wear_img가 제공된 경우)
            emb_prod: 상품 이미지 임베딩 (prod_img가 제공된 경우)
        """
        emb_wear, emb_prod = None, None
        
        if wear_img is not None:
            emb_wear = self.tower_wear(wear_img)   # (B, embed_dim)
        
        if prod_img is not None:
            emb_prod = self.tower_prod(prod_img)   # (B, embed_dim)
            
        if wear_img is not None and prod_img is not None:
            return emb_wear, emb_prod
        elif wear_img is not None:
            return emb_wear
        else:
            return emb_prod
    
    def get_tower_wear(self):
        """착용 이미지 타워만 반환"""
        return self.tower_wear
    
    def get_tower_prod(self):
        """상품 이미지 타워만 반환"""
        return self.tower_prod
    
    def freeze_tower_wear(self, freeze=True):
        """착용 이미지 타워 고정/해제"""
        for param in self.tower_wear.parameters():
            param.requires_grad = not freeze
    
    def freeze_tower_prod(self, freeze=True):
        """상품 이미지 타워 고정/해제"""
        for param in self.tower_prod.parameters():
            param.requires_grad = not freeze 