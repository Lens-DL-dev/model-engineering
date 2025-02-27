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

class HardNegSupConLoss(nn.Module):
    """
    SupConLoss 변형:
      - color_group이 동일하지만 product_code가 다른 경우를
        '하드 네거티브(hard negative)'로 보고, 추가 가중치(alpha)를 부여합니다.
      - alpha > 1.0이면, 하드 네거티브 쌍의 영향이 더 커져,
        모델이 '색상은 같지만 상품이 다른' 경우를 더 강하게 구분하도록 학습합니다.
    """
    def __init__(self, temperature=0.07, alpha=2.0):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, embeddings, product_codes, color_groups):
        """
        Args:
            embeddings: (2B, D) - wearing과 product 임베딩을 concat한 것
            product_codes: list of length B (문자열 상품코드)
            color_groups: list of length B (int 색상 그룹ID; -1인 경우도 있을 수 있음)
        """
        device = embeddings.device
        B = len(product_codes)  # 배치사이즈

        # 1) 임베딩 정규화
        embeddings = F.normalize(embeddings, p=2, dim=1)  # (2B, D)

        # 2) 유사도 행렬 (2B, 2B)
        sim_matrix = torch.matmul(embeddings, embeddings.t()) / self.temperature

        # -----------------------------------------------
        # 3) product_code -> numeric_labels
        #    color_groups -> numeric_colors
        #    SupConLoss에서처럼 각각 2배로 확장(2B)
        # -----------------------------------------------
        # (a) product_code numeric
        unique_codes = {}
        for code in product_codes:
            if code not in unique_codes:
                unique_codes[code] = len(unique_codes)
        numeric_labels = torch.tensor([unique_codes[code] for code in product_codes], device=device)
        labels = numeric_labels.repeat(2)  # (2B,)

        # (b) color_group numeric
        numeric_colors = torch.tensor(color_groups, device=device)
        colors = numeric_colors.repeat(2)  # (2B,)

        # -----------------------------------------------
        # 4) 양성 마스크(동일 product_code) & 하드네거티브 마스크(동일 color_group + 다른 code)
        # -----------------------------------------------
        # (a) same_label_mask
        same_label_mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()  # (2B,2B)
        same_label_mask.fill_diagonal_(0)  # 자기 자신 제외

        # (b) same_color_mask
        same_color_mask = torch.eq(colors.unsqueeze(1), colors.unsqueeze(0)).float()  # (2B,2B)
        same_color_mask.fill_diagonal_(0)

        # 하드네거티브: "color는 같지만 label은 다름"
        hard_neg_mask = same_color_mask * (1.0 - same_label_mask)  # (2B,2B)

        # -----------------------------------------------
        # 5) SupCon 공식 재구성
        #    log_prob = logits - log( sum(exp(logits)) )
        #    하드네거티브 쌍만 exp(logits)에 alpha를 곱해줌
        # -----------------------------------------------
        # (a) for numerical stability
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        # (b) exp(logits)
        exp_logits = torch.exp(logits)  # (2B,2B)
        # -> 하드네거티브 쌍에만 alpha를 곱한다
        exp_logits = exp_logits * (1 + (self.alpha - 1) * hard_neg_mask)

        # (c) 자기 자신 제외
        mask_no_self = 1 - torch.eye(2 * B, device=device)
        exp_logits = exp_logits * mask_no_self

        # (d) log_prob
        denominator = exp_logits.sum(dim=1, keepdim=True) + 1e-12
        log_prob = logits - torch.log(denominator)

        # -----------------------------------------------
        # 6) 양성에 대해서만 평균 log_prob 계산
        #    SupConLoss: mean_log_prob_pos = sum_i( sum_j( mask[i,j]*log_prob[i,j] ) / sum_j(mask[i,j]) )
        # -----------------------------------------------
        mean_log_prob_pos = (same_label_mask * log_prob).sum(dim=1) / (same_label_mask.sum(dim=1) + 1e-12)

        # (e) 전체 평균 (batch-wise)
        loss = -mean_log_prob_pos.mean()
        return loss