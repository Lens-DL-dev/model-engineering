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
    def __init__(self, temperature=0.07, hard_mining=False):
        super().__init__()
        self.temperature = temperature
        self.hard_mining = hard_mining

    def forward(self, embeddings, product_codes, memory_embeddings=None, memory_codes=None):
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


class SimCLRLoss(nn.Module):
    """
    SimCLR Loss (Chen et al., 2020)
    비지도 대조 학습을 위한 손실 함수.
    
    입력: embeddings (2B, D) - 두 view를 concat한 결과
          첫 B개는 첫 번째 augmentation view, 다음 B개는 두 번째 augmentation view
    
    아이디어: (i, i+B)는 positive pair, 나머지 모든 쌍은 negative pair로 간주
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="mean")
        
    def forward(self, embeddings):
        """
        embeddings: 정규화된 임베딩 tensor, shape: [2*batch_size, D]
        """
        device = embeddings.device
        batch_size = embeddings.shape[0] // 2
        
        # similarity matrix 계산
        # sim_matrix[i][j]: i번째 임베딩과 j번째 임베딩의 코사인 유사도
        sim_matrix = torch.mm(embeddings, embeddings.T) / self.temperature
        
        # 자기 자신과의 유사도는 학습에서 제외
        sim_matrix.fill_diagonal_(-float('inf'))
        
        # positive pair의 인덱스를 계산: (i, i+B) 쌍과 (i+B, i) 쌍
        pos_indices = torch.arange(batch_size, device=device)
        pos_indices_1 = pos_indices
        pos_indices_2 = pos_indices + batch_size
        
        # 각 샘플의 positive pair 인덱스
        labels_1 = pos_indices_2  # 첫 번째 batch의 각 샘플에 대한 positive 인덱스는 i+B
        labels_2 = pos_indices_1  # 두 번째 batch의 각 샘플에 대한 positive 인덱스는 i
        
        # 전체 라벨 tensor 구성 (2B,)
        labels = torch.cat([labels_1, labels_2], dim=0)
        
        # loss 계산: CrossEntropy(-log(exp(sim_pos) / sum(exp(sim_all))))
        loss = self.criterion(sim_matrix, labels)
        
        return loss

class MarginSupConLoss(nn.Module):
    def __init__(self, temperature=0.07, margin=0.3, hard_mining=True):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.hard_mining = hard_mining
    
    def forward(self, embeddings, product_codes, memory_embeddings=None, memory_codes=None):
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
        # (2B, ) 크기의 labels 구성
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
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

        # 기본 SupCon 손실
        loss = - mean_log_prob_pos
        loss = loss.mean()
        
        # 마진 기반 페널티 추가
        if self.hard_mining:
            # negative 중 가장 유사한 샘플 찾기
            neg_mask = 1 - mask
            neg_mask = neg_mask * (1 - torch.eye(2 * batch_size, device=device))
            
            # 각 앵커에 대해 가장 가까운 negative 찾기
            sim_matrix_neg = sim_matrix * neg_mask
            sim_matrix_neg[neg_mask == 0] = -float('inf')
            
            hardest_neg_sim, _ = torch.max(sim_matrix_neg, dim=1)
            
            # positive 유사도 평균
            pos_sim = (mask * sim_matrix).sum(dim=1) / (mask.sum(dim=1) + 1e-12)
            
            # 마진 기반 추가 손실
            margin_loss = F.relu(hardest_neg_sim - pos_sim + self.margin)
            
            # 전체 손실에 마진 페널티 추가
            loss = loss + 0.5 * margin_loss.mean()
        
        return loss