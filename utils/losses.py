import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    """
    NT-Xent Loss for SimCLR.
    2B개의 embedding(두 view)을 받아, 각 샘플의 양성(positive) 유사도를 최대화하고
    나머지 negative 샘플과의 거리를 벌입니다.
    
    개선 사항:
      - 각 row에서 양성 유사도가 첫 번째 요소가 되도록 logits를 재구성합니다.
      - fp16 환경에서 -1e9 대신 -1e4를 사용하여 overflow 문제를 방지합니다.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        # 두 view를 이어 붙임 → (2B, D)
        z = torch.cat([z_i, z_j], dim=0)
        z = F.normalize(z, dim=1)
        # 전체 유사도 행렬 (2B, 2B)
        sim = torch.matmul(z, z.T)
        # self-similarity 제거 (fp16에서 -1e9 대신 -1e4 사용)
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim = sim.masked_fill(mask, -1e4)

        # 각 row에 대해, 양성은 (i+B) mod (2B)에 위치
        logits_list = []
        for i in range(2 * batch_size):
            pos_index = (i + batch_size) % (2 * batch_size)
            pos_logit = sim[i, pos_index]  # 스칼라 값
            # self와 양성을 제외한 negative들 선택
            neg_mask = torch.ones(2 * batch_size, dtype=torch.bool, device=z.device)
            neg_mask[i] = False
            neg_mask[pos_index] = False
            neg_logits = sim[i][neg_mask]  # (2B-2,)
            # 양성을 첫 번째 위치에 두도록 logits 구성
            logits_i = torch.cat([pos_logit.unsqueeze(0), neg_logits])  # (2B-1,)
            logits_list.append(logits_i)
        logits = torch.stack(logits_list, dim=0)  # (2B, 2B-1)
        logits /= self.temperature

        # 정답 label은 모두 0 (양성이 첫 번째 위치)
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z.device)
        loss = self.criterion(logits, labels)
        return loss

class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss used in MoCo.
    """
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        return self.criterion(logits, labels)
