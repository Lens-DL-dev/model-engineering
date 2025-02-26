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

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., 2020)
    입력: embeddings (2N, D) – 두 view를 concat한 결과,
         labels: list of product codes (length N) for the original samples.
    내부적으로, labels는 각 샘플에 대해 두 view가 동일하다는 정보를 활용하여,
    같은 클래스(상품 코드)를 가진 샘플들끼리의 similarity를 극대화합니다.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, product_codes):
        """
        embeddings: tensor of shape [2B, D]
        product_codes: list of length B (원래 배치 크기) – 두 view가 동일하므로,
                       이를 2배로 확장하여 [B, B] 비교를 할 수 있음.
        """
        device = embeddings.device
        batch_size = len(product_codes)
        # 두 view가 concat된 임베딩을 정규화
        embeddings = F.normalize(embeddings, p=2, dim=1)
        # similarity 행렬: (2B, 2B)
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # 문자열 product_codes를 숫자 인덱스로 변환
        unique_codes = {}
        for i, code in enumerate(product_codes):
            if code not in unique_codes:
                unique_codes[code] = len(unique_codes)
        
        numeric_labels = torch.tensor([unique_codes[code] for code in product_codes], device=device)
        # 여기서는 간단히, 리스트 내의 동일성 여부를 비교하는 방식을 사용합니다.
        # (2B, ) 크기의 labels를 구성하기 위해, 배치마다 두 view가 있으므로 반복합니다.
        labels = numeric_labels.repeat(2)  # [B] -> [2B]
        
        # mask[i,j] = 1 if labels[i] == labels[j] and i != j
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
        # 자기 자신은 제외 (diagonal 0)
        mask = mask.fill_diagonal_(0)

        # For numerical stability
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        # exp(logits)
        exp_logits = torch.exp(logits) * (1 - torch.eye(2 * batch_size, device=device))
        # log_prob
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # For each sample i, sum over positives j
        # 각 i에 대해, j가 같은 클래스인 경우의 마스크 합
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

        loss = - mean_log_prob_pos
        loss = loss.mean()
        return loss

class TripletLoss(nn.Module):
    """
    TripletMarginLoss를 직접 구현한 예시.
    anchor-positive는 가깝게, anchor-negative는 멀어지도록 유도.
    """
    def __init__(self, margin=1.0, distance_metric='euclidean'):
        super().__init__()
        self.margin = margin
        self.distance_metric = distance_metric

    def forward(self, anchor, positive, negative):
        """
        anchor, positive, negative: 각 (B, D) shape 임베딩
        """
        if self.distance_metric == 'euclidean':
            dist_pos = F.pairwise_distance(anchor, positive, p=2)
            dist_neg = F.pairwise_distance(anchor, negative, p=2)
        else:
            # cosine의 경우, dist = 1 - cos_sim
            dist_pos = 1.0 - F.cosine_similarity(anchor, positive, dim=1)
            dist_neg = 1.0 - F.cosine_similarity(anchor, negative, dim=1)

        # margin + dist_pos - dist_neg 가 0보다 크면 loss 발생
        losses = F.relu(dist_pos - dist_neg + self.margin)
        return losses.mean()