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

def save_tsne_visualization(
    embeddings_list, 
    labels_list, 
    dataset_names=None,
    save_path="tsne_visualization.png", 
    perplexity=30,
    n_components=2,
    random_state=42,
    max_samples_per_dataset=1000,
    title="t-SNE Visualization of Embeddings",
    colors=None
):
    """
    여러 데이터셋(예: 훈련, 검증)의 임베딩을 t-SNE로 2D/3D 시각화합니다.
    
    Parameters:
    -----------
    embeddings_list : list of torch.Tensor
        시각화할 임베딩 텐서들의 리스트 [tensor1, tensor2, ...]
    labels_list : list of list/torch.Tensor
        각 임베딩에 대응되는 라벨들의 리스트 [labels1, labels2, ...]
    dataset_names : list of str, optional
        각 데이터셋의 이름 (예: ['Train', 'Val'])
    save_path : str, default="tsne_visualization.png"
        시각화 결과를 저장할 경로
    perplexity : int, default=30
        t-SNE 알고리즘의 perplexity 파라미터
    n_components : int, default=2
        차원 축소 결과의 차원 수 (2 또는 3)
    random_state : int, default=42
        재현성을 위한 랜덤 시드 값
    max_samples_per_dataset : int, default=1000
        각 데이터셋에서 사용할 최대 샘플 수 (메모리와 계산 속도 제한)
    title : str, default="t-SNE Visualization of Embeddings"
        그래프 제목
    colors : list, optional
        각 데이터셋에 사용할 색상 리스트
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("scikit-learn이 설치되어 있지 않습니다. pip install scikit-learn으로 설치하세요.")
        return
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if dataset_names is None:
        dataset_names = [f"Dataset {i+1}" for i in range(len(embeddings_list))]
    
    if colors is None:
        # 기본 색상 팔레트
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    plt.figure(figsize=(12, 10))
    
    # 모든 임베딩을 하나의 배열로 합치기
    all_embeddings = []
    all_dataset_indices = []  # 각 샘플이 어느 데이터셋에서 왔는지 추적
    all_labels = []
    
    for i, (embeddings, labels) in enumerate(zip(embeddings_list, labels_list)):
        # CPU로 이동하고 NumPy 배열로 변환
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        
        # 각 데이터셋에서 샘플 제한
        n_samples = min(len(embeddings), max_samples_per_dataset)
        indices = np.random.choice(len(embeddings), n_samples, replace=False)
        
        dataset_embeddings = embeddings[indices]
        dataset_labels = np.array(labels)[indices] if len(labels) > 0 else np.zeros(n_samples)
        
        all_embeddings.append(dataset_embeddings)
        all_dataset_indices.extend([i] * n_samples)
        all_labels.extend(dataset_labels)
    
    # 모든 임베딩 병합
    all_embeddings = np.vstack(all_embeddings)
    
    # t-SNE 적용
    print(f"Running t-SNE on {len(all_embeddings)} samples...")
    tsne = TSNE(n_components=n_components, perplexity=perplexity, 
                n_iter=1000, random_state=random_state)
    embeddings_tsne = tsne.fit_transform(all_embeddings)
    
    # 각 데이터셋을 다른 색상으로 표시
    all_dataset_indices = np.array(all_dataset_indices)
    all_labels = np.array(all_labels)
    
    if n_components == 2:
        # 2D 시각화
        for i, dataset_name in enumerate(dataset_names):
            mask = (all_dataset_indices == i)
            plt.scatter(
                embeddings_tsne[mask, 0], 
                embeddings_tsne[mask, 1],
                color=colors[i % len(colors)],
                label=dataset_name,
                alpha=0.7,
                s=30,
                edgecolors='none'
            )
        
        plt.title(title, fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.xlabel("t-SNE dimension 1", fontsize=14)
        plt.ylabel("t-SNE dimension 2", fontsize=14)
    
    else:  # 3D 시각화
        from mpl_toolkits.mplot3d import Axes3D
        ax = plt.axes(projection='3d')
        
        for i, dataset_name in enumerate(dataset_names):
            mask = (all_dataset_indices == i)
            ax.scatter(
                embeddings_tsne[mask, 0], 
                embeddings_tsne[mask, 1], 
                embeddings_tsne[mask, 2],
                color=colors[i % len(colors)],
                label=dataset_name,
                alpha=0.7,
                s=30,
                edgecolors='none'
            )
        
        ax.set_title(title, fontsize=16)
        ax.legend(fontsize=12)
        ax.set_xlabel("t-SNE dimension 1", fontsize=14)
        ax.set_ylabel("t-SNE dimension 2", fontsize=14)
        ax.set_zlabel("t-SNE dimension 3", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"t-SNE visualization saved to {save_path}")
    
    # 3D 대화형 HTML로 저장 (선택적)
    if n_components == 3:
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            
            fig = go.Figure()
            
            # 데이터셋별 표시
            for i, dataset_name in enumerate(dataset_names):
                mask = (all_dataset_indices == i)
                
                fig.add_trace(go.Scatter3d(
                    x=embeddings_tsne[mask, 0],
                    y=embeddings_tsne[mask, 1],
                    z=embeddings_tsne[mask, 2],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=colors[i % len(colors)],
                        opacity=0.7
                    ),
                    name=dataset_name
                ))
            
            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title="t-SNE dimension 1",
                    yaxis_title="t-SNE dimension 2",
                    zaxis_title="t-SNE dimension 3",
                ),
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                ),
                margin=dict(l=0, r=0, b=0, t=30)
            )
            
            html_path = save_path.replace('.png', '_interactive.html')
            fig.write_html(html_path)
            print(f"Interactive 3D t-SNE visualization saved to {html_path}")
        except ImportError:
            print("Plotly가 설치되어 있지 않아 대화형 3D 시각화를 생성할 수 없습니다.")
    
    return embeddings_tsne, all_dataset_indices, all_labels
