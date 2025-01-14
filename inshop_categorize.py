import torch
from PIL import Image
import numpy as np
import json
import os
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from collections import Counter
import multiprocessing
from functools import partial
from tqdm import tqdm

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model and processor
processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = model.to(device)

# 카테고리 매핑 정의
category_mapping = {
    0: "background",
    1: "hat",
    4: "main_top",
    7: "main_top",
    5: "bottom",
    6: "bottom",
    9: "shoes",
    10: "shoes",
}

def process_batch(image_paths, batch_size=4):
    results = []
    
    # 배치 단위로 처리
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        images = []
        image_arrays = []
        product_ids = []
        
        for image_path in batch_paths:
            image = Image.open(image_path).convert('RGB')
            product_id = os.path.basename(os.path.dirname(image_path))
            
            images.append(image)
            image_arrays.append(np.array(image))
            product_ids.append(product_id)

        # 배치 처리
        inputs = processor(images=image_arrays, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            # 즉시 GPU 메모리 해제
            del outputs
            torch.cuda.empty_cache()

        for idx, logits_single in enumerate(logits):
            upsampled_logits = torch.nn.functional.interpolate(
                logits_single.unsqueeze(0),
                size=images[idx].size[::-1],
                mode="bilinear",
                align_corners=False,
            )
            
            pred_seg = torch.argmax(upsampled_logits, dim=1)[0]
            
            # 픽셀 수 기준으로 가장 많이 등장한 카테고리 찾기
            pixel_counts = Counter()
            for old_idx, category in category_mapping.items():
                if category != "background":
                    mask = (pred_seg == old_idx)
                    pixel_counts[category] += mask.sum().item()
            
            if pixel_counts:
                dominant_category = max(pixel_counts.items(), key=lambda x: x[1])[0]
                results.append({
                    "productId": product_ids[idx],
                    "category": dominant_category
                })
            
            # 메모리 정리
            del upsampled_logits, pred_seg
            torch.cuda.empty_cache()
        
        # 배치 처리 후 메모리 정리
        del inputs, logits
        torch.cuda.empty_cache()
    
    return results

def process_images_parallel(data_type="train"):
    base_path = f"./data/{data_type}/product_images"
    output_path = f"./data/{data_type}/product_category.json"
    
    # 모든 제품 이미지 경로 수집
    image_paths = []
    for product_id in os.listdir(base_path):
        front_image = os.path.join(base_path, product_id, f"{product_id}_F.jpg")
        if os.path.exists(front_image):
            image_paths.append(front_image)
    
    print(f"Found {len(image_paths)} images to process")
    
    # 워커 수를 줄임
    num_workers = 2  # 워커 수를 2로 제한
    print(f"Using {num_workers} workers")
    
    # 전체 이미지를 워커 수만큼 분할
    chunks = np.array_split(image_paths, num_workers)
    batches = []
    for chunk in chunks:
        batch_size = 4  # 배치 크기
        for i in range(0, len(chunk), batch_size):
            batches.append(chunk[i:i + batch_size])
    
    # 병렬 처리
    all_results = []
    with multiprocessing.Pool(num_workers) as pool:
        for batch_results in tqdm(
            pool.imap(process_batch, batches),
            total=len(batches),
            desc=f"Processing {data_type} images"
        ):
            all_results.extend(batch_results)
    
    # 결과 저장
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Results saved to {output_path}")
    return len(all_results)

if __name__ == '__main__':
    # multiprocessing 시작 방식을 'spawn'으로 설정
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass  # 이미 설정된 경우 무시
    
    # GPU 메모리 캐시 정리
    torch.cuda.empty_cache()
    
    # train과 val 데이터 모두 처리
    for data_type in ["train", "val"]:
        print(f"\nProcessing {data_type} dataset...")
        processed_count = process_images_parallel(data_type)
        print(f"Processed {processed_count} images for {data_type} dataset") 