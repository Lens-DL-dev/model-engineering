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

def compute_topk_accuracy(query_emb, candidate_emb, topk=(1, 5, 10)):
    """
    query_emb, candidate_emb: (N, D) 정규화된 임베딩
    각 query의 정답은 동일 인덱스에 있다고 가정합니다.
    Returns:
      dict: {1: top1_accuracy, 5: top5_accuracy, 10: top10_accuracy} (단위: %)
    """
    # 차원 확인 및 처리
    if query_emb.dim() == 1:
        query_emb = query_emb.unsqueeze(0)
    if candidate_emb.dim() == 1:
        candidate_emb = candidate_emb.unsqueeze(0)
    
    # 임베딩 정규화
    query_emb = F.normalize(query_emb, p=2, dim=1)
    candidate_emb = F.normalize(candidate_emb, p=2, dim=1)
    
    # (N, N) 유사도 행렬 (내적 사용)
    similarity = torch.matmul(query_emb, candidate_emb.t())
    
    # 차원 처리
    if similarity.dim() == 0:
        similarity = similarity.unsqueeze(0).unsqueeze(0)
    elif similarity.dim() == 1:
        similarity = similarity.unsqueeze(0)
    
    max_k = max(topk)
    # max_k가 후보 수를 초과하지 않도록 보장
    max_k = min(max_k, candidate_emb.size(0))
    
    _, indices = similarity.topk(max_k, dim=1, largest=True, sorted=True)
    # 각 query의 정답은 자신의 인덱스에 있다고 가정
    gt = torch.arange(query_emb.size(0), device=query_emb.device).unsqueeze(1)
    correct = indices.eq(gt)
    
    topk_acc = {}
    for k in topk:
        k = min(k, max_k)  # k가 max_k를 초과하지 않도록 보장
        topk_acc[k] = correct[:, :k].float().sum().item() / query_emb.size(0) * 100.0
    
    return topk_acc