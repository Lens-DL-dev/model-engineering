# utils/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    """
    NT-Xent Loss for SimCLR.
    두 view의 embedding을 입력받아 배치 내 positive 쌍의 cosine similarity를 최대화합니다.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0)  # (2B, D)
        z = F.normalize(z, dim=1)
        similarity_matrix = torch.matmul(z, z.T)  # (2B, 2B)
        
        # Mask out self-similarity with a smaller value for float16 compatibility
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        similarity_matrix = similarity_matrix.masked_fill(mask, -100.0)
        
        # Positive similarities: i-th sample in first half with (i+batch_size)-th sample
        positives = torch.cat([torch.diag(similarity_matrix, batch_size), 
                              torch.diag(similarity_matrix, -batch_size)]).view(2 * batch_size, 1)
        
        logits = similarity_matrix / self.temperature
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z.device)
        loss = self.criterion(logits, labels)
        return loss

class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss used in MoCo.
    이미 모델 내에서 logits과 정답 label(0번 위치가 positive)으로 구성된 상태에서 사용합니다.
    """
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, logits, labels):
        return self.criterion(logits, labels)
