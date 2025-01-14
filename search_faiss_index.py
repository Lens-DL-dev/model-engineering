import os
import sys
import json
import argparse

import numpy as np
import faiss
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt

from PIL import Image
from faiss.contrib import torch_utils  # GPU 연동 유틸

# 내부 모듈
sys.path.append('.')
from models.efficientnet_v2 import EfficientNetV2L

def load_model(checkpoint_path, embed_dim=512, device='cuda'):
    model = EfficientNetV2L(pretrained=False, embed_dim=embed_dim)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    return model

def extract_embedding(model, image_path, transform, device='cuda'):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(image_tensor)
    return embedding.squeeze(0).cpu().numpy()

def load_faiss_index(index_path, use_gpu=True):
    # CPU index 불러오기
    cpu_index = faiss.read_index(index_path)

    if use_gpu:
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        return gpu_index
    else:
        return cpu_index

def visualize_topk_results(query_image_path, topk_product_ids, distances, product_imgs_dir, out_path=None):
    # Query 이미지를 포함해서 subplot 생성
    fig, axes = plt.subplots(1, len(topk_product_ids) + 1, figsize=(5 * (len(topk_product_ids) + 1), 5))
    if len(topk_product_ids) + 1 == 1:
        axes = [axes]

    # Query 이미지 표시
    query_img = Image.open(query_image_path).convert('RGB')
    axes[0].imshow(query_img)
    axes[0].set_title('Query Image')
    axes[0].axis('off')

    # 검색된 상품 이미지들 표시
    for ax, pid, dist in zip(axes[1:], topk_product_ids, distances):
        f_path = os.path.join(product_imgs_dir, pid, f"{pid}_F.jpg")
        if not os.path.exists(f_path):
            candidates = [p for p in os.listdir(os.path.join(product_imgs_dir, pid)) if p.endswith('.jpg')]
            if candidates:
                f_path = os.path.join(product_imgs_dir, pid, candidates[0])

        if os.path.exists(f_path):
            img = Image.open(f_path).convert('RGB')
            ax.imshow(img)
            # 메타 정보를 포함한 타이틀 표시
            title = f"ID: {pid}\nDist: {dist:.4f}"
            ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/last_ckpt.pth.tar')
    parser.add_argument('--faiss_dir', type=str, default='./faiss_index')
    parser.add_argument('--product_imgs_dir', type=str, default='./data/train/product_images')
    parser.add_argument('--segmented_wearing_image', type=str, required=True)
    parser.add_argument('--brand', type=str, required=True)
    parser.add_argument('--category', type=str, required=True)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--output_json', type=str, default='search_result.json')
    parser.add_argument('--viz_out', type=str, default='search_visual.png')
    parser.add_argument('--use_gpu_faiss', action='store_true',
                        help='Use FAISS GPU index for search.')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True

    # ------------------------------
    # 1) Load meta
    # ------------------------------
    index_path = os.path.join(args.faiss_dir, 'product.faiss')
    embeddings_path = os.path.join(args.faiss_dir, 'embeddings.npy')
    product_ids_path = os.path.join(args.faiss_dir, 'product_ids.npy')
    brand_map_path = os.path.join(args.faiss_dir, 'brand_map.json')
    category_map_path = os.path.join(args.faiss_dir, 'category_map.json')

    if not all(os.path.exists(p) for p in [index_path, embeddings_path, product_ids_path,
                                           brand_map_path, category_map_path]):
        print("[Error] Required files are missing.")
        return

    emb_array = np.load(embeddings_path)
    product_ids = np.load(product_ids_path).astype(str).tolist()

    with open(brand_map_path, 'r') as f:
        pid_to_brand = json.load(f)
    with open(category_map_path, 'r') as f:
        pid_to_category = json.load(f)

    pid_to_index = {pid: i for i, pid in enumerate(product_ids)}

    # ------------------------------
    # 2) Load FAISS index (GPU)
    # ------------------------------
    print("[Info] Loading FAISS index...")
    faiss_index = load_faiss_index(index_path, use_gpu=args.use_gpu_faiss)
    print(f"[Info] Index total size: {faiss_index.ntotal}")

    # ------------------------------
    # 3) Load model
    # ------------------------------
    model = load_model(args.checkpoint, embed_dim=args.embed_dim, device=device)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    # ------------------------------
    # 4) Query embedding
    # ------------------------------
    query_emb = extract_embedding(model, args.segmented_wearing_image, transform, device=device)
    query_emb = query_emb.astype(np.float32)[np.newaxis, :]  # (1, dim)

    # ------------------------------
    # 5) Filter by brand & category
    # ------------------------------
    candidate_indices = []
    for pid in product_ids:
        if pid_to_brand.get(pid, "") == args.brand and pid_to_category.get(pid, "") == args.category:
            candidate_indices.append(pid_to_index[pid])

    if not candidate_indices:
        print(f"[Info] No candidate product for brand='{args.brand}', category='{args.category}'")
        return

    # 간단히 SubIndex -> GPU
    sub_cpu_index = faiss.IndexFlatL2(args.embed_dim)  # CPU
    sub_cpu_index.add(emb_array[candidate_indices].astype(np.float32))

    if args.use_gpu_faiss:
        res = faiss.StandardGpuResources()
        sub_gpu_index = faiss.index_cpu_to_gpu(res, 0, sub_cpu_index)
        distances, indices = sub_gpu_index.search(query_emb, args.topk)
    else:
        distances, indices = sub_cpu_index.search(query_emb, args.topk)

    distances = distances[0]
    indices = indices[0]

    # SubIndex 내 인덱스 -> 실제 product_ids 인덱스
    retrieved_pids = [product_ids[candidate_indices[sub_i]] for sub_i in indices]

    # ------------------------------
    # 6) 결과 JSON
    # ------------------------------
    results = []
    for rank, (pid, dist) in enumerate(zip(retrieved_pids, distances), start=1):
        results.append({
            "rank": rank,
            "productId": pid,
            "brand": pid_to_brand.get(pid, ""),
            "category": pid_to_category.get(pid, ""),
            "distance": float(dist),
        })

    with open(args.output_json, 'w') as f:
        json.dump({"results": results}, f, indent=4)
    print(f"[Info] Result saved: {args.output_json}")

    # ------------------------------
    # 7) 시각화
    # ------------------------------
    visualize_topk_results(
        query_image_path=args.segmented_wearing_image,  # Query 이미지 경로 추가
        topk_product_ids=retrieved_pids,
        distances=distances,
        product_imgs_dir=args.product_imgs_dir,
        out_path=args.viz_out
    )
    print(f"[Info] Visualization saved: {args.viz_out}")


if __name__ == '__main__':
    main()
