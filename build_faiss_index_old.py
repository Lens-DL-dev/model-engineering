import os
import sys
import json
import glob
import argparse

import numpy as np
import torch
import faiss
from faiss.contrib import torch_utils  # GPU 사용 시 torch GPU -> FAISS GPU 연동 유틸
from PIL import Image
from tqdm import tqdm

# timm, etc
import torchvision.transforms as T

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/last_ckpt.pth.tar')
    parser.add_argument('--image_dir', type=str, default='./data/train/product_images')
    parser.add_argument('--brands_json', type=str, default='./data/example/brands.json')
    parser.add_argument('--category_json', type=str, default='./data/train/product_category.json')
    parser.add_argument('--faiss_out_dir', type=str, default='./faiss_index')
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--force_rebuild', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.faiss_out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True  # GPU 성능 최적화

    # -----------------------------
    # 1) Load brand, category info
    # -----------------------------
    with open(args.brands_json, 'r') as f:
        brand_info = json.load(f)

    with open(args.category_json, 'r') as f:
        product_category = json.load(f)

    productid_to_brand = {}
    for b_name, p_ids in brand_info.items():
        for pid in p_ids:
            productid_to_brand[pid] = b_name

    productid_to_category = {}
    for item in product_category:
        pid = item['productId']
        cat = item['category']
        productid_to_category[pid] = cat

    # -----------------------------
    # 2) Load model
    # -----------------------------
    print("Loading model checkpoint...")
    model = load_model(args.checkpoint, embed_dim=args.embed_dim, device=device)

    # -----------------------------
    # 3) Transform
    # -----------------------------
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    # -----------------------------
    # 4) Embedding Extraction
    # -----------------------------
    print("Collecting product images...")
    image_paths = glob.glob(os.path.join(args.image_dir, '**', '*_F.jpg'), recursive=True)

    embeddings_path = os.path.join(args.faiss_out_dir, 'embeddings.npy')
    product_ids_path = os.path.join(args.faiss_out_dir, 'product_ids.npy')

    emb_list = []
    product_ids = []
    if (not args.force_rebuild) and os.path.exists(embeddings_path) and os.path.exists(product_ids_path):
        print("[Info] Found existing embeddings. Loading them...")
        emb_list = np.load(embeddings_path)
        product_ids = np.load(product_ids_path).astype(str).tolist()

    existing_product_set = set(product_ids)

    for img_path in tqdm(image_paths, desc="Embedding..."):
        product_folder = os.path.basename(os.path.dirname(img_path))  # => productId
        if product_folder in existing_product_set:
            continue
        if product_folder not in productid_to_brand or product_folder not in productid_to_category:
            continue

        emb = extract_embedding(model, img_path, transform, device=device)
        if len(emb_list) == 0:
            emb_list = emb[np.newaxis, :]
        else:
            emb_list = np.vstack([emb_list, emb])
        product_ids.append(product_folder)
        existing_product_set.add(product_folder)

    if isinstance(emb_list, np.ndarray) and len(product_ids) > 0:
        np.save(embeddings_path, emb_list)
        np.save(product_ids_path, np.array(product_ids))
        print(f"[Info] Saved embeddings: shape={emb_list.shape}, # of products={len(product_ids)}")
    else:
        print("[Info] No new embeddings. Exiting.")
        return

    # -----------------------------
    # 5) Save brand/category maps
    # -----------------------------
    brand_map_path = os.path.join(args.faiss_out_dir, 'brand_map.json')
    category_map_path = os.path.join(args.faiss_out_dir, 'category_map.json')
    with open(brand_map_path, 'w') as f:
        json.dump(productid_to_brand, f, indent=4)
    with open(category_map_path, 'w') as f:
        json.dump(productid_to_category, f, indent=4)

    # -----------------------------
    # 6) Build FAISS Index on GPU
    # -----------------------------
    embed_dim = emb_list.shape[1]

    # 1) CPU IndexFlatL2 생성
    cpu_index = faiss.IndexFlatL2(embed_dim)

    # 2) GPU 메모리 할당
    #    (T4 16GB 고려 시, 여러 백만 개 embedding이 아니라면 IndexFlatL2 + GPU로도 충분)
    res = faiss.StandardGpuResources()

    # 3) CPU index -> GPU index 복사
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

    # 4) 실제 벡터 추가
    gpu_index.add(emb_list.astype(np.float32))

    print(f"[Info] GPU index total size: {gpu_index.ntotal}")

    # 5) 완성된 GPU index를 다시 CPU index로 복사 후 디스크 저장
    final_index = faiss.index_gpu_to_cpu(gpu_index)
    faiss_index_path = os.path.join(args.faiss_out_dir, 'product.faiss')
    faiss.write_index(final_index, faiss_index_path)
    print(f"[Info] FAISS index saved: {faiss_index_path}")


if __name__ == '__main__':
    main()
