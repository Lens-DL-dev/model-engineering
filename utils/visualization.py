# utils/visualization.py
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch.nn.functional as F

def save_contrastive_matrix(
    emb_anchor, emb_candidate, labels, 
    save_path="contrastive_matrix.png",
    distance_metric='euclidean',
    margin=None
):
    """
    거리를 산점도로 시각화 + margin 선
    Positive(blue circle), Negative(red x)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if distance_metric == 'euclidean':
        dist = F.pairwise_distance(emb_anchor, emb_candidate, p=2)
    else:
        dist = 1.0 - F.cosine_similarity(emb_anchor, emb_candidate, dim=1)

    dist_data = dist.detach().cpu().numpy()
    label_data = labels.detach().cpu().numpy()
    x_axis = np.arange(len(dist_data))

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


def save_topk_image_samples_with_mask(
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

    anchor_cpu = anchor_imgs.cpu()
    anchor_image = anchor_cpu[:,:3,:,:]
    anchor_mask = anchor_cpu[:,3,:,:]
    candidate_cpu = candidate_imgs.cpu()
    candidate_image = candidate_cpu[:,:3,:,:]
    candidate_mask = candidate_cpu[:,3,:,:]
    dist_cpu = dist.detach().cpu().numpy()

    for rank in range(min(k, anchor_cpu.size(0))):
        idx = sorted_indices[rank].item()
        a_img = anchor_image[idx]
        a_mask = anchor_mask[idx][None, :, :].repeat(3, 1, 1)
        c_img = candidate_image[idx]
        c_mask = candidate_mask[idx][None, :, :].repeat(3, 1, 1)
        d_val = dist_cpu[idx]

        grid = torchvision.utils.make_grid([a_img, c_img, a_mask, c_mask], nrow=2, padding=5, normalize=True)
        # to PIL
        pil_img = torchvision.transforms.ToPILImage()(grid)
        out_name = f"top_{rank+1}_dist_{d_val:.3f}.jpg"
        pil_img.save(os.path.join(out_dir, out_name))
