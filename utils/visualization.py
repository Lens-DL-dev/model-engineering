import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch.nn.functional as F

def save_contrastive_matrix(
    emb_anchor, emb_positive, labels, 
    save_path="contrastive_matrix.png",
    distance_metric='euclidean',
    margin=None
):
    """
    두 view (anchor와 positive) 간의 임베딩 거리를 산점도로 시각화합니다.
    모든 anchor-positive 쌍에 대한 거리를 표시합니다 (배치 내 모든 조합 포함).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    batch_size = emb_anchor.size(0)
    
    # 모든 가능한 쌍에 대한 거리 계산
    if distance_metric == 'euclidean':
        # [b1, d] x [b2, d] -> [b1, b2]의 거리 행렬
        dist_matrix = torch.cdist(emb_anchor, emb_positive, p=2)
    else:  # cosine
        # [b, d] -> [b, d] (normalize)
        norm_anchor = F.normalize(emb_anchor, dim=1)
        norm_positive = F.normalize(emb_positive, dim=1)
        # [b1, d] x [b2, d] -> [b1, b2]의 유사도 행렬
        similarity = torch.mm(norm_anchor, norm_positive.t())
        dist_matrix = 1.0 - similarity
    
    # 대각선: positive pair (같은 인덱스끼리의 쌍)
    diag_indices = torch.arange(batch_size)
    pos_distances = dist_matrix[diag_indices, diag_indices].detach().cpu().numpy()
    
    # 비대각선: negative pair (다른 인덱스끼리의 쌍)
    mask = torch.ones_like(dist_matrix, dtype=torch.bool)
    mask[diag_indices, diag_indices] = False
    neg_distances = dist_matrix[mask].detach().cpu().numpy()
    
    # 시각화
    plt.figure(figsize=(10, 6))
    
    # x축 데이터 준비
    x_pos = np.arange(len(pos_distances))
    x_neg = np.random.uniform(0, len(pos_distances), size=len(neg_distances))
    
    # positive pair와 negative pair 구분해서 플롯
    plt.scatter(x_pos, pos_distances, color='blue', marker='o', alpha=0.7, label='Positive Pairs')
    plt.scatter(x_neg, neg_distances, color='red', marker='.', alpha=0.2, label='Negative Pairs')
    
    if margin is not None:
        plt.axhline(y=margin, color='gray', linestyle='--', label=f'Margin={margin}')

    plt.title("Contrastive Distance Distribution (All Pairs)")
    plt.xlabel("Sample Index")
    plt.ylabel("Distance")
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def save_topk_image_samples(
    anchor_imgs, positive_imgs, emb_anchor, emb_positive, 
    k=3, distance_metric='euclidean', out_dir="vis_samples"
):
    """
    두 view (anchor, positive)의 이미지와 임베딩을 받아, 거리 기준 상위 k개 샘플을 이미지로 시각화합니다.
    """
    os.makedirs(out_dir, exist_ok=True)
    if distance_metric == 'euclidean':
        dist = F.pairwise_distance(emb_anchor, emb_positive, p=2)
    else:
        dist = 1.0 - F.cosine_similarity(emb_anchor, emb_positive, dim=1)

    sorted_indices = torch.argsort(dist)
    anchor_cpu = anchor_imgs.cpu()[:,:3,:,:]
    positive_cpu = positive_imgs.cpu()[:,:3,:,:]
    dist_cpu = dist.detach().cpu().numpy()

    for rank in range(min(k, anchor_cpu.size(0))):
        idx = sorted_indices[rank].item()
        a_img = anchor_cpu[idx]
        p_img = positive_cpu[idx]
        d_val = dist_cpu[idx]
        grid = torchvision.utils.make_grid([a_img, p_img], nrow=2, padding=5, normalize=True)
        pil_img = torchvision.transforms.ToPILImage()(grid)
        out_name = f"top_{rank+1}_dist_{d_val:.3f}.jpg"
        pil_img.save(os.path.join(out_dir, out_name))
