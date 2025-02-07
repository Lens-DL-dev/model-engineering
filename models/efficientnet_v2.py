# models/efficientnet_v2.py
import torch
import torch.nn as nn
import torch.nn.init as init
import timm

class EmbeddingHead(nn.Module):
    def __init__(self, backbone_out, embed_dim):
        super().__init__()
        self.embedding_head = nn.Sequential(
            nn.Linear(backbone_out, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self._initialize_weights()  # 추가된 초기화 함수

    def _initialize_weights(self):
        # Kaiming He Initialization
        for m in self.embedding_head.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                init.ones_(m.weight)
                init.zeros_(m.bias)
    
    def forward(self, x):
        return self.embedding_head(x)

class EfficientNetV2L(nn.Module):
    """
    tf_efficientnetv2_l.in21k -> backbone_out = 1280 (고정)
    Contrastive Learning을 위한 임베딩 레이어 포함
    """
    def __init__(self, pretrained=True, embed_dim=512, in_channels=4):
        super().__init__()
        # 1) 모델 생성
        self.backbone = timm.create_model(
            "tf_efficientnetv2_l.in21k",
            pretrained=pretrained,
            num_classes=0,  # 최종 FC 제거
            in_chans=in_channels # 250120_kdi 기존 3채널 + Segmentation 마스크 채널
        )
        # 2) num_features = 1280 고정 가정
        backbone_out = self.backbone.num_features  # 혹은 self.backbone.num_features 로 확인해도 무방
        
        # (선택) gradient checkpointing 적용:
        # if hasattr(self.backbone, 'blocks'):
        #     from torch.utils.checkpoint import checkpoint
        #     original_blocks = self.backbone.blocks
            
        #     def blocks_with_checkpoint(*args, **kwargs):
        #         return checkpoint(original_blocks, *args, **kwargs)
            
        #     self.backbone.blocks = blocks_with_checkpoint

        # 3) 임베딩 레이어
        self.embedding_head = EmbeddingHead(backbone_out, embed_dim)

    def forward(self, x):
        features = self.backbone(x)         # [B, 1280]
        embeddings = self.embedding_head(features)  # [B, embed_dim]
        return embeddings