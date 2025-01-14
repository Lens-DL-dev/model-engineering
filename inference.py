import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from models.efficientnet_v2 import efficientnet_v2_l
from models.vit import vit_base
from models.segmentation import FacereSegmentationModel
from utils.config import load_config
from utils.faiss_helper import search_similar
from sklearn.cluster import KMeans
import json
import argparse

def extract_color_info(image, num_clusters=5):
    # 이미지를 numpy 배열로 변환
    img_array = np.array(image)
    img_array = img_array.reshape((-1, 3))
    # KMeans를 사용하여 주요 색상 추출
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(img_array)
    dominant_color = kmeans.cluster_centers_[0]
    # RGB 값을 0-1 사이로 정규화
    dominant_color = dominant_color / 255.0
    return dominant_color  # (R, G, B) 형태의 numpy 배열 반환

def load_model(model_name, checkpoint_path, additional_feature_dim):
    if model_name == 'efficientnet':
        model = efficientnet_v2_l(use_additional_features=True, additional_feature_dim=additional_feature_dim)
    elif model_name == 'vit':
        model = vit_base()
    else:
        raise ValueError('지원하지 않는 모델입니다.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 체크포인트 로드
    if os.path.isfile(checkpoint_path):
        print(f"=> 체크포인트 '{checkpoint_path}' 로드 중")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"=> 체크포인트 '{checkpoint_path}' 로드 완료")
    else:
        print(f"=> '{checkpoint_path}' 에서 체크포인트를 찾을 수 없습니다.")
        exit()

    model.eval()
    return model

def get_transform(model_name):
    if model_name == 'efficientnet':
        # EfficientNetV2-L의 입력 크기와 변환
        image_size = (384, 384)
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std =[0.5, 0.5, 0.5])
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            normalize
        ])
    elif model_name == 'vit':
        # ViT의 입력 크기와 변환
        from transformers import ViTImageProcessor
        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        def vit_transform(image):
            return processor(images=image, return_tensors='pt')['pixel_values'].squeeze(0)
        transform = vit_transform
    else:
        raise ValueError('지원하지 않는 모델입니다.')
    return transform

def main():
    parser = argparse.ArgumentParser(description='Fashion Similarity Inference')
    parser.add_argument('--config', default='config.yaml', type=str, help='Path to config file')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--brand_name', type=str, required=True, help='Brand name to search within')
    parser.add_argument('--output_path', type=str, default='results.json', help='Path to save results')
    parser.add_argument('--save_intermediate', action='store_true', help='Save intermediate outputs for visualization')
    args = parser.parse_args()

    config = load_config(args.config)

    # 설정 불러오기
    model_name = config['model']['name']
    checkpoint_path = config['model']['checkpoint']
    embeddings_dir = config['data']['embeddings']
    index_path = config['faiss']['index_path']
    dimension = config['faiss']['dimension']
    additional_feature_dim = 4  # 카테고리(1) + 색상(3)

    # 모델 로드
    model = load_model(model_name, checkpoint_path, additional_feature_dim)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 세그멘테이션 모델 로드
    segmentation_model = FacereSegmentationModel(model_path='models/facere_plus.onnx')

    # 이미지 로드
    image = Image.open(args.image_path).convert('RGB')

    # 세그멘테이션 적용
    segmented_image, category = segmentation_model.segment(image)

    # 카테고리 매핑 (세그멘테이션 모델과 동일한 매핑 사용)
    category_mapping = {
        'top': 0,
        'bottom': 1,
        'shoes': 2,
        # 필요한 카테고리 매핑 추가
    }
    category_idx = category_mapping.get(category, -1)

    # 색상 정보 추출
    color_info = extract_color_info(segmented_image)
    color_info = torch.tensor(color_info, dtype=torch.float32)

    if args.save_intermediate:
        # 세그멘테이션 결과 이미지 저장
        segmented_image.save('segmented_image.png')
        print("세그멘테이션 결과 이미지가 'segmented_image.png'로 저장되었습니다.")

        # 색상 정보 저장
        with open('color_info.json', 'w') as f:
            color_data = {
                'category': category,
                'category_idx': category_idx,
                'color': color_info.tolist()
            }
            json.dump(color_data, f)
            print("색상 정보가 'color_info.json'으로 저장되었습니다.")

    # 이미지 전처리
    transform = get_transform(model_name)
    input_tensor = transform(segmented_image).unsqueeze(0).to(device)

    # 추가 특징 정보 생성
    additional_features = torch.cat((torch.tensor([category_idx], dtype=torch.float32), color_info), dim=0).unsqueeze(0).to(device)

    # 임베딩 생성
    with torch.no_grad():
        embedding = model(input_tensor, additional_features)
    embedding_np = embedding.cpu().detach().numpy()

    # FAISS 검색
    ids = []
    brands = [args.brand_name]
    meta_data = []
    if not os.path.exists(index_path):
        from utils.faiss_helper import build_faiss_index
        print("FAISS 인덱스가 존재하지 않습니다. 인덱스를 생성합니다...")
        ids, meta_data = build_faiss_index(embeddings_dir, index_path, dimension, brand_filter=brands)
    else:
        # 메타데이터 로드
        # 예: embeddings_dir 내의 파일 리스트를 사용
        brand_dir = os.path.join(embeddings_dir, args.brand_name)
        embeddings_files = os.listdir(brand_dir)
        ids = [f"{args.brand_name}/{file}" for file in embeddings_files]

    # 유사한 이미지 검색
    results, distances = search_similar(embedding_np, index_path, ids, top_k=5)

    # 결과 출력 및 저장
    print("가장 유사한 이미지 결과:")
    results_data = []
    for idx, (result, distance) in enumerate(zip(results, distances)):
        print(f"Top {idx+1}: {result}, Distance: {distance:.4f}")
        result_item = {
            'rank': idx + 1,
            'image_path': result,
            'distance': float(distance)
        }
        results_data.append(result_item)

    # 결과를 JSON 파일로 저장
    with open(args.output_path, 'w') as f:
        json.dump({'results': results_data}, f, indent=4)
        print(f"결과가 '{args.output_path}'로 저장되었습니다.")

if __name__ == '__main__':
    main()