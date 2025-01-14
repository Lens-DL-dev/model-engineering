# utils/metrics.py
import torch
import torch.nn.functional as F

def calculate_accuracy_contrastive(emb1, emb2, labels, threshold=1.0, distance_metric='euclidean'):
    """
    emb1, emb2 : (B, embed_dim)
    labels     : (B,)  1 => positive / 0 => negative
    threshold  : euclidean 거리 기준으로 positive/negative 판단
    """
    if distance_metric == 'euclidean':
        dist = F.pairwise_distance(emb1, emb2, p=2)
        # dist < threshold -> positive로 예측
        preds = (dist < threshold).long()
    else:
        # cosine distance: dist = 1 - cos_sim
        # dist < threshold -> (1 - cos_sim) < threshold -> cos_sim > (1 - threshold)
        dist = 1.0 - F.cosine_similarity(emb1, emb2, dim=1)
        preds = (dist < threshold).long()

    correct = (preds == labels.long()).sum().item()
    total = labels.size(0)
    acc = (correct / total) * 100.0
    return acc
