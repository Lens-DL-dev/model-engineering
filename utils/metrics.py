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


def calculate_tp_fp_tn_fn(emb1, emb2, labels, threshold=1.0, distance_metric='euclidean'):
    """
    emb1, emb2 : (B, embed_dim)
    labels     : (B,)  1 => positive / 0 => negative
    threshold  : euclidean 거리 기준으로 positive/negative 판단
    """
    if distance_metric == 'euclidean':
        dist = F.pairwise_distance(emb1, emb2, p=2)
        preds = (dist < threshold).long()
    else:
        dist = 1.0 - F.cosine_similarity(emb1, emb2, dim=1)
        preds = (dist < threshold).long()
    
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    
    return tp, fp, tn, fn


def compute_f1_score(tp: int, fp: int, tn: int, fn: int) -> float:
    """
    TP, FP, TN, FN 값을 이용해 F1-score를 계산하는 함수.

    Args:
        tp (int): True Positives
        fp (int): False Positives
        tn (int): True Negatives (사용되지 않음)
        fn (int): False Negatives

    Returns:
        float: F1-score (0.0 ~ 1.0)
    """
    # Precision = TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Recall = TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F1-score = 2 * (Precision * Recall) / (Precision + Recall)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1
