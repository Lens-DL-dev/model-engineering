# utils/visualization.py
import os
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import torchvision

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
        grid_x = vutils.make_grid(x[:num_samples], nrow=4, padding=2, normalize=True)
        grid_y = vutils.make_grid(y[:num_samples], nrow=4, padding=2, normalize=True)
        
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

def save_contrastive_matrix(
    emb_anchor, emb_candidate, labels=None, 
    save_path="contrastive_matrix.png",
    distance_metric='euclidean',
    margin=None
):
    """
    거리를 산점도로 시각화 + margin 선
    Positive(blue circle), Negative(red x)
    
    Parameters:
        emb_anchor: (B, D) 임베딩 텐서
        emb_candidate: (B, D) 임베딩 텐서
        labels: None이면 대각선(같은 인덱스)을 positive, 다른 것을 negative로 가정
               아니면 주어진 labels을 사용 (1=positive, 0=negative)
        save_path: 저장 경로
        distance_metric: 'euclidean' 또는 'cosine'
        margin: 표시할 margin 값
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if distance_metric == 'euclidean':
        dist = F.pairwise_distance(emb_anchor, emb_candidate, p=2)
    else:
        dist = 1.0 - F.cosine_similarity(emb_anchor, emb_candidate, dim=1)

    dist_data = dist.detach().cpu().numpy()
    x_axis = np.arange(len(dist_data))
    
    # 레이블이 None이면 대각선(같은 인덱스)을 positive로 가정
    if labels is None:
        label_data = np.zeros(len(dist_data))
        # 대각선 요소(같은 인덱스)만 1로 설정
        for i in range(len(dist_data)):
            label_data[i] = 1  # 같은 인덱스는 positive
    else:
        label_data = labels.detach().cpu().numpy()

    pos_mask = (label_data == 1)
    neg_mask = (label_data == 0)

    plt.figure(figsize=(8, 5))
    # Positive
    plt.scatter(x_axis[pos_mask], dist_data[pos_mask],
                color='blue', marker='o', alpha=0.7, label='Positive (label=1)')
    # Negative
    plt.scatter(x_axis[neg_mask], dist_data[neg_mask],
                color='red', marker='x', alpha=0.7, label='Negative (label=0)')

    if margin is not None:
        plt.axhline(y=margin, color='gray', linestyle='--', label=f'Margin={margin}')

    plt.title("Contrastive Distance Scatter")
    plt.xlabel("Sample Index")
    plt.ylabel("Distance")
    plt.grid(True)
    plt.legend(loc='best')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def save_topk_image_samples(
    anchor_imgs, candidate_imgs, emb_anchor, emb_candidate, 
    k=3, distance_metric='euclidean', out_dir="vis_samples"
):
    """
    anchor_imgs, candidate_imgs: (B, C, H, W)
    emb_anchor, emb_candidate: (B, dim)
    => 거리 기준으로 가장 가까운 상위 k개 샘플을 이미지로 시각화
    """
    os.makedirs(out_dir, exist_ok=True)

    if distance_metric == 'euclidean':
        dist = F.pairwise_distance(emb_anchor, emb_candidate, p=2)
    else:
        dist = 1.0 - F.cosine_similarity(emb_anchor, emb_candidate, dim=1)

    sorted_indices = torch.argsort(dist)  # 오름차순 (가까운 순)

    anchor_cpu = anchor_imgs.cpu()[:,:3,:,:] # 250122_wsj Mask 채널 제거
    candidate_cpu = candidate_imgs.cpu()[:,:3,:,:] # 250122_wsj Mask 채널 제거
    dist_cpu = dist.detach().cpu().numpy()

    for rank in range(min(k, anchor_cpu.size(0))):
        idx = sorted_indices[rank].item()
        a_img = anchor_cpu[idx]
        c_img = candidate_cpu[idx]
        d_val = dist_cpu[idx]

        grid = torchvision.utils.make_grid([a_img, c_img], nrow=2, padding=5, normalize=True)
        # to PIL
        pil_img = torchvision.transforms.ToPILImage()(grid)
        out_name = f"top_{rank+1}_dist_{d_val:.3f}.jpg"
        pil_img.save(os.path.join(out_dir, out_name))



