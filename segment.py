import torch
from PIL import Image
import numpy as np
import json
import os
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import shutil
import multiprocessing
from torch.utils.data import DataLoader
from functools import partial
from tqdm import tqdm

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model and processor
processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = model.to(device)

# 새로운 카테고리 매핑 정의
category_mapping = {
    0: "background",  # Background
    1: "hat",        # Hat
    4: "main_top",   # Upper-clothes
    7: "main_top",   # Dress
    5: "bottom",     # Skirt
    6: "bottom",     # Pants
    9: "shoes",      # Left-shoe
    10: "shoes",     # Right-shoe
}

new_category_to_idx = {
    "background": 0,
    "hat": 1,
    "main_top": 2,
    "inner_top": 3,
    "bottom": 4,
    "shoes": 5
}

new_id2label = {
    0: "background",
    1: "hat",
    2: "main_top",
    3: "inner_top",
    4: "bottom",
    5: "shoes"
}

def remap_segmentation(pred_seg):
    new_seg = torch.zeros_like(pred_seg)
    for old_idx, category in category_mapping.items():
        mask = (pred_seg == old_idx)
        new_seg[mask] = new_category_to_idx[category]
    return new_seg

# 설정 변수들
wearing_info_path = "./data/train/wearing_info.json"  # 사용자 지정 가능하도록 변수로 분리
masked_dir = "./data/train/segmented_images"
meta_dir = "./data/train/meta_info"

# wearing_info.json 로드 및 처리
def load_wearing_info(json_path):
    with open(json_path, 'r') as f:
        wearing_info = json.load(f)
    # 이미지 파일명을 키로 하는 딕셔너리로 변환하여 검색 속도 향상
    return {item['wearing']: item for item in wearing_info}

def process_batch(image_paths, wearing_info_dict, batch_size=4):
    images = []
    image_arrays = []
    base_names = []
    
    for image_path in image_paths:
        image = Image.open(image_path).convert('RGB')
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        images.append(image)
        image_arrays.append(np.array(image))
        base_names.append(base_name)

    # 배치 처리
    inputs = processor(images=image_arrays, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # 배치 전체의 결과를 메모리에 모아서 한 번에 저장
    batch_results = []
    
    for idx, (image, logits_single) in enumerate(zip(images, logits)):
        base_name = base_names[idx]
        image_info = wearing_info_dict.get(os.path.basename(image_paths[idx]))
        
        if not image_info:  # wearing_info에 없는 이미지는 건너뛰기
            continue

        # Upsample logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits_single.unsqueeze(0),
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        
        probabilities = torch.nn.functional.softmax(upsampled_logits, dim=1)
        confidence, pred_seg = torch.max(probabilities, dim=1)
        pred_seg = pred_seg[0]
        confidence = confidence[0]

        remapped_seg = remap_segmentation(pred_seg)
        
        metadata = []
        save_tasks = []  # 저장할 이미지 정보를 모아서 한 번에 처리

        for class_idx in torch.unique(remapped_seg):
            class_name = new_id2label[int(class_idx)]
            if class_name == "background":
                continue
                
            # wearing_info 기반 필터링
            if (class_name == "hat" and image_info["hat"] is None) or \
               (class_name == "bottom" and image_info["bottom"] is None) or \
               (class_name == "shoes" and image_info["shoes"] is None) or \
               (class_name == "main_top" and image_info["main_top"] is None and image_info["inner_top"] is None):
                continue

            class_mask = (remapped_seg == class_idx).cpu().numpy()
            masked = np.array(image) * np.expand_dims(class_mask, axis=2)
            white_bg = np.ones_like(masked) * 255
            masked = np.where(masked == 0, white_bg, masked)
            
            output_path = f"{masked_dir}/{base_name}_{class_name}.png"
            class_confidence = float(confidence[class_mask].mean().item())
            
            save_tasks.append((output_path, masked))
            metadata.append({
                "class": class_name,
                "path": f"masked/{base_name}_{class_name}.png",
                "acc": class_confidence
            })
        
        if metadata:  # 메타데이터가 있는 경우만 저장
            batch_results.append((f"{meta_dir}/{base_name}.json", metadata))
            # 이미지 ���장 작업 실행
            for output_path, masked_img in save_tasks:
                Image.fromarray(masked_img.astype(np.uint8)).save(output_path)

    # 메타데이터 일괄 저장
    for json_path, metadata in batch_results:
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=4)

def process_images_parallel(image_paths, wearing_info_dict):
    # CPU 코어 수 확인 (전체 코어의 75%를 사용)
    num_workers = max(1, int(multiprocessing.cpu_count() * 0.75))
    print(f"Using {num_workers} workers")
    batch_size = int(64 / num_workers)
    
    # 전체 이미지를 워커 수만큼 분할
    chunks = np.array_split(image_paths, num_workers)
    
    # 모든 배치를 하나의 리스트로 생성
    all_batches = []
    for chunk in chunks:
        batches = [chunk[i:i + batch_size] for i in range(0, len(chunk), batch_size)]
        all_batches.extend(batches)
    
    # 모든 배치를 한 번에 병렬 처리
    with multiprocessing.Pool(num_workers) as pool:
        process_batch_partial = partial(process_batch, wearing_info_dict=wearing_info_dict)
        list(tqdm(
            pool.imap(process_batch_partial, all_batches),
            total=len(all_batches),
            desc=f"Processing images"
        ))

# Main execution
if __name__ == '__main__':
    # multiprocessing 시작 방식을 'spawn'으로 설정
    multiprocessing.set_start_method('spawn')
    
    # GPU 메모리 캐시 정리
    torch.cuda.empty_cache()
    
    # 디렉토리 초기화
    for dir_path in [masked_dir, meta_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)

    # wearing_info 로드
    wearing_info_dict = load_wearing_info(wearing_info_path)

    print("Starting image processing...")
    tmp_dir = "./unzipped/train/wearing_images"
    image_paths = [
        os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    print(f"Found {len(image_paths)} images to process")

    process_images_parallel(image_paths, wearing_info_dict)
    print("Processing completed!")