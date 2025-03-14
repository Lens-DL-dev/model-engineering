import os
import sys
import json
import argparse

import numpy as np
import faiss
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from transformers import CLIPVisionModel
from PIL import Image

def load_model(checkpoint_path, device='cuda'):
    model = CLIPVisionModel.from_pretrained(checkpoint_path)
    model.to(device)
    model.eval()
    return model

def extract_embedding(model, image_path, transform, device='cuda'):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        embedding = outputs.pooler_output
    return embedding.squeeze(0).cpu().numpy()

def load_faiss_index(index_path, use_gpu=True):
    cpu_index = faiss.read_index(index_path)
    try:
        if use_gpu:
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            return gpu_index
    except AttributeError:
        print("[Info] GPU FAISS not available, using CPU version")
    return cpu_index

def calculate_accuracy(distances):
    """Calculate accuracy based on distance"""
    max_dist = np.max(distances)
    accuracies = 1 - (distances / max_dist)  # 거리가 멀수록 정확도가 낮아짐
    return accuracies

def visualize_topk_results(query_image_path, topk_product_ids, distances, accuracies, product_imgs_dir, out_path=None):
    fig, axes = plt.subplots(1, len(topk_product_ids) + 1, figsize=(5 * (len(topk_product_ids) + 1), 5))
    if len(topk_product_ids) + 1 == 1:
        axes = [axes]

    query_img = Image.open(query_image_path).convert('RGB')
    axes[0].imshow(query_img)
    axes[0].set_title('Query Image')
    axes[0].axis('off')

    for ax, pid, dist, acc in zip(axes[1:], topk_product_ids, distances, accuracies):
        img_path = os.path.join(product_imgs_dir, f"{pid}.jpg")
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
            ax.imshow(img)
            title = f"ID: {pid}\nDist: {dist:.4f}\nAcc: {acc:.2%}"
            ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/latest')
    parser.add_argument('--faiss_dir', type=str, default='./faiss_index')
    parser.add_argument('--product_imgs_dir', type=str, default='/home/ubuntu/dev/dataset/main/product')
    parser.add_argument('--query_image', type=str, required=True)
    parser.add_argument('--brand', type=str, required=True)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--output_json', type=str, default='search_result.json')
    parser.add_argument('--viz_out', type=str, default='search_visual.png')
    parser.add_argument('--use_gpu_faiss', action='store_true')
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

    if not all(os.path.exists(p) for p in [index_path, embeddings_path, product_ids_path, brand_map_path]):
        print("[Error] Required files are missing.")
        return

    emb_array = np.load(embeddings_path)
    product_ids = np.load(product_ids_path).astype(str).tolist()

    with open(brand_map_path, 'r') as f:
        pid_to_brand = json.load(f)

    pid_to_index = {pid: i for i, pid in enumerate(product_ids)}

    # ------------------------------
    # 2) Load FAISS index
    # ------------------------------
    print("[Info] Loading FAISS index...")
    faiss_index = load_faiss_index(index_path, use_gpu=args.use_gpu_faiss)
    print(f"[Info] Index total size: {faiss_index.ntotal}")

    # ------------------------------
    # 3) Load model & transform - CLIP 모델에 맞게 설정
    # ------------------------------
    model = load_model(args.checkpoint, device=device)
    transform = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225])
    ])

    # ------------------------------
    # 4) Query embedding
    # ------------------------------
    query_emb = extract_embedding(model, args.query_image, transform, device=device)
    query_emb = query_emb.astype(np.float32)[np.newaxis, :]

    # ------------------------------
    # 5) Filter by brand
    # ------------------------------
    candidate_indices = []
    for pid in product_ids:
        if pid_to_brand.get(pid, "") == args.brand:
            candidate_indices.append(pid_to_index[pid])

    if not candidate_indices:
        print(f"[Info] No candidate product for brand='{args.brand}'")
        return

    sub_cpu_index = faiss.IndexFlatL2(emb_array.shape[1])
    sub_cpu_index.add(emb_array[candidate_indices].astype(np.float32))

    try:
        if args.use_gpu_faiss:
            res = faiss.StandardGpuResources()
            sub_gpu_index = faiss.index_cpu_to_gpu(res, 0, sub_cpu_index)
            distances, indices = sub_gpu_index.search(query_emb, args.topk)
        else:
            distances, indices = sub_cpu_index.search(query_emb, args.topk)
    except AttributeError:
        distances, indices = sub_cpu_index.search(query_emb, args.topk)

    distances = distances[0]
    indices = indices[0]
    accuracies = calculate_accuracy(distances)

    retrieved_pids = [product_ids[candidate_indices[sub_i]] for sub_i in indices]

    # ------------------------------
    # 6) Results
    # ------------------------------
    results = []
    for rank, (pid, dist, acc) in enumerate(zip(retrieved_pids, distances, accuracies), start=1):
        # Skip results with accuracy less than 10%
        if acc < 0.1:
            continue
            
        results.append({
            "rank": rank,
            "productId": pid,
            "brand": pid_to_brand.get(pid, ""),
            "distance": float(dist),
            "accuracy": float(acc)
        })

    # If no results remain after filtering
    if not results:
        print("[Info] No results found with accuracy >= 10%")

    with open(args.output_json, 'w') as f:
        json.dump({"results": results}, f, indent=4)
    print(f"[Info] Result saved: {args.output_json}")

    # ------------------------------
    # 7) Visualization
    # ------------------------------
    visualize_topk_results(
        query_image_path=args.query_image,
        topk_product_ids=retrieved_pids,
        distances=distances,
        accuracies=accuracies,
        product_imgs_dir=args.product_imgs_dir,
        out_path=args.viz_out
    )
    print(f"[Info] Visualization saved: {args.viz_out}")

if __name__ == '__main__':
    main()
