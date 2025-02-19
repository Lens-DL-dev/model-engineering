import os
import sys
import json
import glob
import argparse

import numpy as np
import torch
import faiss
from transformers import CLIPVisionModel  # EfficientNet -> CLIP 변경
from PIL import Image
from tqdm import tqdm

# timm, etc
import torchvision.transforms as T

# 내부 모듈
sys.path.append('.')
from models.efficientnet_v2 import EfficientNetV2L

def load_model(checkpoint_path, device='cuda'):
    # CLIP 모델 로드로 변경
    model = CLIPVisionModel.from_pretrained(checkpoint_path)
    model.to(device)
    model.eval()
    return model

def extract_embedding(model, image_path, transform, device='cuda'):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        # CLIP의 pooled output 사용
        embedding = outputs.pooler_output
    return embedding.squeeze(0).cpu().numpy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/latest')
    parser.add_argument('--image_dir', type=str, default='/home/ubuntu/dev/dataset/main/product')
    parser.add_argument('--brands_json', type=str, default='/home/ubuntu/dev/dataset/main/brandInfo.json')
    parser.add_argument('--faiss_out_dir', type=str, default='./faiss_index')
    parser.add_argument('--force_rebuild', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.faiss_out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True  # GPU 성능 최적화

    # -----------------------------
    # 1) Load brand info
    # -----------------------------
    with open(args.brands_json, 'r') as f:
        brand_info = json.load(f)

    productid_to_brand = {}
    for b_name, p_ids in brand_info.items():
        for pid in p_ids:
            productid_to_brand[pid] = b_name

    # -----------------------------
    # 2) Load model
    # -----------------------------
    print("Loading model checkpoint...")
    model = load_model(args.checkpoint, device=device)

    # -----------------------------
    # 3) Transform - CLIP 모델에 맞게 설정
    # -----------------------------
    transform = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225])
    ])

    # -----------------------------
    # 4) Embedding Extraction
    # -----------------------------
    print("Collecting product images...")
    image_paths = glob.glob(os.path.join(args.image_dir, '**', '*.jpg'), recursive=True)
    print(f"[Info] Found {len(image_paths)} images")
    
    # brand_info 디버깅
    print(f"[Info] Number of brands: {len(brand_info)}")
    print(f"[Info] Total products in brand_info: {sum(len(products) for products in brand_info.values())}")
    
    embeddings_path = os.path.join(args.faiss_out_dir, 'embeddings.npy')
    product_ids_path = os.path.join(args.faiss_out_dir, 'product_ids.npy')

    emb_list = []
    product_ids = []
    
    if not args.force_rebuild and os.path.exists(embeddings_path) and os.path.exists(product_ids_path):
        print("[Info] Found existing embeddings. Loading them...")
        emb_list = np.load(embeddings_path)
        product_ids = np.load(product_ids_path).astype(str).tolist()
        if len(emb_list) == 0:
            print("[Info] Existing embeddings are empty. Starting fresh...")
            emb_list = []
            product_ids = []

    existing_product_set = set(product_ids)
    print(f"[Info] Found {len(existing_product_set)} existing products")

    skipped_products = 0
    new_products = 0
    for img_path in tqdm(image_paths, desc="Embedding..."):
        # 파일 이름에서 제품 ID 추출 (확장자 제외)
        product_id = os.path.splitext(os.path.basename(img_path))[0]
        
        # 디버깅을 위한 첫 몇 개의 제품 정보 출력
        if new_products == 0 and skipped_products < 5:
            print(f"[Debug] Processing image: {img_path}")
            print(f"[Debug] Product ID: {product_id}")
            print(f"[Debug] In brand_info: {product_id in productid_to_brand}")
            skipped_products += 1

        if not args.force_rebuild and product_id in existing_product_set:
            continue
            
        if product_id not in productid_to_brand:
            continue

        try:
            emb = extract_embedding(model, img_path, transform, device=device)
            if len(emb_list) == 0:
                emb_list = emb[np.newaxis, :]
            else:
                emb_list = np.vstack([emb_list, emb])
            product_ids.append(product_id)
            existing_product_set.add(product_id)
            new_products += 1
            
            # 첫 번째 성공적인 임베딩 후 정보 출력
            if new_products == 1:
                print(f"[Info] First successful embedding shape: {emb.shape}")
                
        except Exception as e:
            print(f"[Error] Failed to process {img_path}: {str(e)}")

    print(f"[Info] Processed {new_products} new products")

    if isinstance(emb_list, np.ndarray) and len(product_ids) > 0:
        print(f"[Info] Final embeddings shape: {emb_list.shape}")
        np.save(embeddings_path, emb_list)
        np.save(product_ids_path, np.array(product_ids))
        print(f"[Info] Saved embeddings: shape={emb_list.shape}, # of products={len(product_ids)}")
    else:
        print("[Info] No new embeddings. Exiting.")
        return

    # -----------------------------
    # 5) Save brand map
    # -----------------------------
    brand_map_path = os.path.join(args.faiss_out_dir, 'brand_map.json')
    with open(brand_map_path, 'w') as f:
        json.dump(productid_to_brand, f, indent=4)

    # -----------------------------
    # 6) Build FAISS Index
    # -----------------------------
    embed_dim = emb_list.shape[1]
    print(f"[Info] Building FAISS index with dimension {embed_dim}")

    try:
        # GPU 버전 시도
        res = faiss.StandardGpuResources()
        cpu_index = faiss.IndexFlatL2(embed_dim)
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        gpu_index.add(emb_list.astype(np.float32))
        print(f"[Info] GPU index total size: {gpu_index.ntotal}")
        final_index = faiss.index_gpu_to_cpu(gpu_index)
    except AttributeError:
        # GPU 버전이 없으면 CPU 버전 사용
        print("[Info] GPU FAISS not available, using CPU version instead")
        final_index = faiss.IndexFlatL2(embed_dim)
        final_index.add(emb_list.astype(np.float32))
        print(f"[Info] CPU index total size: {final_index.ntotal}")

    faiss_index_path = os.path.join(args.faiss_out_dir, 'product.faiss')
    faiss.write_index(final_index, faiss_index_path)
    print(f"[Info] FAISS index saved: {faiss_index_path}")


if __name__ == '__main__':
    main()
