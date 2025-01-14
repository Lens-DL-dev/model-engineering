# utils/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    label=1 => 두 임베딩은 가까워야
    label=0 => 두 임베딩은 margin 이상 떨어져야
    """
    def __init__(self, margin=1.0, distance_metric='euclidean'):
        super().__init__()
        self.margin = margin
        self.distance_metric = distance_metric

    def forward(self, emb1, emb2, labels):
        if self.distance_metric == 'euclidean':
            dist = F.pairwise_distance(emb1, emb2, p=2)
        else:
            dist = 1.0 - F.cosine_similarity(emb1, emb2, dim=1)
        
        # (dist)^2 for positives
        pos_loss = labels * 0.5 * dist.pow(2)
        # margin-based penalty for negatives
        neg_loss = (1 - labels) * 0.5 * F.relu(self.margin - dist).pow(2)
        loss = pos_loss + neg_loss
        return loss.mean()
