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
    A100 GPU에 최적화된 Supervised Contrastive Loss
    메모리 효율성 개선과 하드 마이닝 기능 강화
    """
    def __init__(self, temperature=0.07, base_temperature=0.07, hard_mining=True, adaptive_temperature=True):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.hard_mining = hard_mining
        self.adaptive_temperature = adaptive_temperature

    def forward(self, embeddings, product_codes, memory_embeddings=None, memory_codes=None):
        """
        embeddings: tensor of shape [2B, D]
        product_codes: list of length B (원래 배치 크기)
        memory_embeddings: Optional tensor of shape [M, D] (메모리 뱅크 임베딩)
        memory_codes: Optional list of length M (메모리 뱅크 상품 코드)
        """
        device = embeddings.device
        batch_size = len(product_codes)
        
        # 두 view가 concat된 임베딩을 정규화
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # 문자열 product_codes를 숫자 인덱스로 변환
        unique_codes = {}
        for i, code in enumerate(product_codes):
            if code not in unique_codes:
                unique_codes[code] = len(unique_codes)
        
        numeric_labels = torch.tensor([unique_codes[code] for code in product_codes], device=device)
        # 여기서는 간단히, 리스트 내의 동일성 여부를 비교하는 방식을 사용합니다.
        # (2B, ) 크기의 labels를 구성하기 위해, 배치마다 두 view가 있으므로 반복합니다.
        labels = numeric_labels.repeat(2)  # [B] -> [2B]
        
        # 메모리 뱅크가 제공된 경우 활용
        if memory_embeddings is not None and memory_codes is not None and len(memory_codes) > 0:
            # 메모리 코드도 같은 unique_codes 딕셔너리로 변환
            memory_numeric = []
            for code in memory_codes:
                if code not in unique_codes:
                    unique_codes[code] = len(unique_codes)
                memory_numeric.append(unique_codes[code])
            
            memory_labels = torch.tensor(memory_numeric, device=device)
            
            # 원래 임베딩과 메모리 임베딩 결합
            all_embeddings = torch.cat([embeddings, memory_embeddings], dim=0)
            all_labels = torch.cat([labels, memory_labels], dim=0)
            
            # 원본 임베딩만 앵커로 사용 (2B 개)
            anchor_idx = torch.arange(2*batch_size, device=device)
            anchor_count = len(anchor_idx)
            
            # 모든 앵커와 모든 샘플(자신 포함) 간의 유사도 계산
            # 메모리 효율성을 위해 청크 단위로 계산 (대용량 메모리 뱅크 사용 시)
            chunk_size = min(2048, all_embeddings.shape[0])  # 청크 크기 설정
            sim_matrix = torch.zeros((2*batch_size, all_embeddings.shape[0]), device=device)
            
            for i in range(0, 2*batch_size, chunk_size):
                end_i = min(i + chunk_size, 2*batch_size)
                for j in range(0, all_embeddings.shape[0], chunk_size):
                    end_j = min(j + chunk_size, all_embeddings.shape[0])
                    # 부분 유사도 행렬 계산
                    sim_matrix[i:end_i, j:end_j] = torch.matmul(
                        embeddings[i:end_i], all_embeddings[j:end_j].T
                    ) / self.temperature
        else:
            # 원래 방식대로 batch 내에서만 계산
            sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
            all_labels = labels
            anchor_idx = torch.arange(2*batch_size, device=device)
            anchor_count = len(anchor_idx)
        
        # For numerical stability
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = sim_matrix - logits_max.detach()
        
        # 각 앵커에 대해 마스크 생성 (같은 클래스 = positive)
        anchor_labels = all_labels[anchor_idx].view(-1, 1)
        mask = (anchor_labels == all_labels.view(1, -1)).float()
        
        # 자기 자신은 제외
        if memory_embeddings is None:
            # 메모리가 없는 경우 대각선만 0으로
            mask = mask.fill_diagonal_(0)
        else:
            # 메모리가 있는 경우 원본 anchor 간의 대각선만 0으로
            for i in range(anchor_count):
                mask[i, i] = 0
        
        # exp(sim) 계산, 자기 자신은 제외
        exp_sim = torch.exp(sim_matrix)
        if memory_embeddings is None:
            # 자기 자신과의 유사도는 exp 계산에서 제외
            exp_sim = exp_sim * (1 - torch.eye(2 * batch_size, device=device))
        
        # 분모: 모든 negative + positive 샘플의 exp_sim 합
        # 분자: positive 샘플의 exp_sim 합
        # log(분자/분모) = log(분자) - log(분모)
        denominator = exp_sim.sum(dim=1, keepdim=True)
        
        # 하드 네거티브 마이닝을 위한 온도 조절
        if self.hard_mining and self.adaptive_temperature:
            # 각 앵커에 대해 가장 어려운 negative 샘플 식별
            # mask의 반전: 1이면 negative, 0이면 positive
            neg_mask = 1 - mask
            
            # 자기 자신은 무시
            if memory_embeddings is None:
                neg_mask = neg_mask * (1 - torch.eye(2 * batch_size, device=device))
            
            # negative 중에서 가장 유사도가 높은 샘플 찾기 (어려운 negative)
            sim_matrix_neg = sim_matrix * neg_mask
            sim_matrix_neg[neg_mask == 0] = -float('inf')  # negative 아닌 것은 -inf로 설정
            hardest_neg_sim, _ = torch.max(sim_matrix_neg, dim=1)
            
            # positive와 negative 유사도 사이의 마진 기반 온도 조절
            pos_sim = (mask * sim_matrix).sum(dim=1) / (mask.sum(dim=1) + 1e-12)
            margin = pos_sim - hardest_neg_sim
            
            # 마진이 작을수록 (어려운 샘플) 온도를 낮춰 더 선명하게 분리
            # 마진이 큰 경우 (쉬운 샘플) 온도를 높여 덜 공격적으로 학습
            adaptive_temp = self.temperature * torch.sigmoid(margin).unsqueeze(1)
            sim_matrix = sim_matrix / adaptive_temp
            
            # 하드 네거티브에 대해 가중치 부여 (더 어려운 negative에 더 집중)
            neg_weight = F.softmax(sim_matrix_neg, dim=1)
            neg_weight = neg_weight * neg_mask
            # 이제 exp_sim에 neg_weight를 곱해서 하드 네거티브에 더 집중
            exp_sim = torch.exp(sim_matrix)
            if memory_embeddings is None:
                exp_sim = exp_sim * (1 - torch.eye(2 * batch_size, device=device))
            exp_sim = exp_sim * (mask + neg_weight * neg_mask)
            denominator = exp_sim.sum(dim=1, keepdim=True)
        
        # 각 앵커에 대한 positive 샘플의 log(exp_sim)
        log_prob = sim_matrix - torch.log(denominator + 1e-12)
        
        # 각 앵커에 대한 positive 샘플 간의 평균 log_prob
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-12)
        
        # 손실 계산 (negated mean_log_prob_pos)
        loss = - (self.base_temperature / self.temperature) * mean_log_prob_pos.mean()
        
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