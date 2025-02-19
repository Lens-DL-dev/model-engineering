# utils/visualization.py
import os
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

def save_visualization(x, y, loss_value, vis_path, mode="batch"):
    """
    시각화 이미지를 저장하는 함수.

    Parameters:
        x: mode에 따라 달라짐
           - mode="batch": (B, C, H, W) 형태의 착용 이미지 텐서 (예: 첫 번째 view)
           - mode="epoch": (N, D) 형태의 query 임베딩 텐서
        y: mode에 따라 달라짐
           - mode="batch": (B, C, H, W) 형태의 상품 이미지 텐서 (예: 두 번째 view)
           - mode="epoch": (N, D) 형태의 candidate 임베딩 텐서
        loss_value: 현재 loss 값 (float) – 시각화 타이틀에 표시
        vis_path: 저장할 이미지 파일 경로
        mode: "batch" 또는 "epoch"
    """
    if mode == "batch":
        # 배치 모드: x와 y는 이미지 텐서 (B, C, H, W)
        # 8개 정도의 이미지를 grid로 만들어서 시각화
        num_samples = min(x.size(0), 8)
        grid_x = vutils.make_grid(x[:num_samples], nrow=4, padding=2)
        grid_y = vutils.make_grid(y[:num_samples], nrow=4, padding=2)
        
        # Tensor -> numpy (C,H,W) -> (H,W,C)
        np_grid_x = grid_x.cpu().numpy().transpose(1, 2, 0)
        np_grid_y = grid_y.cpu().numpy().transpose(1, 2, 0)
        
        # 시각화: 두 이미지를 나란히 배치
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(np.clip(np_grid_x, 0, 1))
        axes[0].axis('off')
        axes[0].set_title("Wearing Images")
        
        axes[1].imshow(np.clip(np_grid_y, 0, 1))
        axes[1].axis('off')
        axes[1].set_title("Product Images")
        
        fig.suptitle(f"Batch Visualization - Loss: {loss_value:.4f}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        os.makedirs(os.path.dirname(vis_path), exist_ok=True)
        plt.savefig(vis_path)
        plt.close(fig)
    
    elif mode == "epoch":
        # epoch 모드: x와 y는 각각 (N, D) 임베딩 텐서
        # cosine similarity 행렬을 계산하여 시각화
        x_norm = F.normalize(x, dim=1)
        y_norm = F.normalize(y, dim=1)
        similarity = torch.matmul(x_norm, y_norm.t()).cpu().numpy()
        
        plt.figure(figsize=(8, 6))
        plt.imshow(similarity, cmap='viridis', interpolation='nearest')
        plt.title(f"Epoch Similarity Matrix - Loss: {loss_value:.4f}")
        plt.colorbar()
        plt.xlabel("Candidate Index")
        plt.ylabel("Query Index")
        plt.tight_layout()
        os.makedirs(os.path.dirname(vis_path), exist_ok=True)
        plt.savefig(vis_path)
        plt.close()
    else:
        raise ValueError(f"Unknown mode '{mode}' for save_visualization")
