import os
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import torchvision.transforms as transforms
from models.efficientnet_v2 import EfficientNetV2L

# 세그멘테이션 관련 설정
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

test_num = 0

def remap_segmentation(pred_seg):
    new_seg = torch.zeros_like(pred_seg)
    for old_idx, category in category_mapping.items():
        mask = (pred_seg == old_idx)
        new_seg[mask] = new_category_to_idx[category]
    return new_seg

def segment_image(image_path, seg_model, processor, device):
    """이미지 세그멘테이션 수행"""
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=np.array(image), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = seg_model(**inputs)
        logits = outputs.logits

    # Upsample logits
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )
    
    probabilities = torch.nn.functional.softmax(upsampled_logits, dim=1)
    confidence, pred_seg = torch.max(probabilities, dim=1)
    pred_seg = pred_seg[0]
    confidence = confidence[0]
    remapped_seg = remap_segmentation(pred_seg)
    
    segmented_images = {}
    for class_idx in torch.unique(remapped_seg):
        class_name = new_id2label[int(class_idx)]
        if class_name == "background":
            continue
            
        class_mask = (remapped_seg == class_idx).cpu().numpy()
        masked = np.array(image) * np.expand_dims(class_mask, axis=2)
        white_bg = np.ones_like(masked) * 255
        masked = np.where(masked == 0, white_bg, masked)
        
        segmented_images[class_name] = Image.fromarray(masked.astype(np.uint8))
    
    return segmented_images

def get_latest_checkpoint(checkpoint_dir):
    """가장 최신 체크포인트 찾기"""
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth.tar')]
    if not checkpoints:
        raise FileNotFoundError("체크포인트를 찾을 수 없습니다.")
    
    latest = max(checkpoints, key=lambda x: int(x.split('_')[1].split('.')[0]))
    return os.path.join(checkpoint_dir, latest)

def load_model(checkpoint_path, device):
    """모델 로드 및 체크포인트 적용"""
    model = EfficientNetV2L(pretrained=False, embed_dim=512).to(device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

def get_embedding(image_path, model, device, image_size=224):
    """이미지의 임베딩 추출"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = model(image_tensor)
    
    return embedding, image

def calculate_distance(emb1, emb2, metric='euclidean'):
    """두 임베딩 간의 거리 계산"""
    if metric == 'euclidean':
        return torch.norm(emb1 - emb2, p=2)
    elif metric == 'cosine':
        return 1 - torch.nn.functional.cosine_similarity(emb1, emb2)
    else:
        raise ValueError(f"지원하지 않는 거리 메트릭: {metric}")

def visualize_comparison(wearing_img, inshop_img, distance, save_path):
    """두 이미지와 거리값을 시각화"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(wearing_img)
    plt.title('Wearing Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(inshop_img)
    plt.title('Inshop Image')
    plt.axis('off')
    
    plt.suptitle(f'Distance: {distance:.4f}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 세그멘테이션 모델 준비
    processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    seg_model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
    seg_model = seg_model.to(device)
    seg_model.eval()
    
    # 2. 임베딩 모델 준비
    latest_ckpt = get_latest_checkpoint('./checkpoints')
    model = load_model(latest_ckpt, device)
    
    # 3. 디렉토리 생성
    os.makedirs('./segmented', exist_ok=True)
    os.makedirs('./results', exist_ok=True)
    
    # 4. wearing 이미지 세그멘테이션
    wearing_path = f'./test/test_{test_num}_wearing.jpg'
    segmented_images = segment_image(wearing_path, seg_model, processor, device)
    
    # 세그먼트된 이미지 저장
    for class_name, seg_img in segmented_images.items():
        seg_img.save(f'./segmented/test_{test_num}_wearing_{class_name}.png')
    
    # 5. inshop 이미지 처리
    inshop_path = f'./test/test_{test_num}_inshop.jpg'
    inshop_embedding, inshop_img = get_embedding(inshop_path, model, device)
    
    # 6. 각 세그먼트된 이미지에 대해 거리 계산 및 시각화
    for class_name, seg_img in segmented_images.items():
        # 임시 저장
        temp_path = f'./segmented/temp_{test_num}_{class_name}.png'
        seg_img.save(temp_path)
        
        # 임베딩 추출
        seg_embedding, _ = get_embedding(temp_path, model, device)
        
        # 거리 계산
        distance = calculate_distance(seg_embedding, inshop_embedding)
        
        # 시각화 및 저장
        visualize_comparison(
            seg_img,
            inshop_img,
            distance.item(),
            f'./results/comparison_{test_num}_{class_name}.png'
        )
        
        os.remove(temp_path)  # 임시 파일 삭제

if __name__ == '__main__':
    main()
