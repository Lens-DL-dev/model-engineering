import os
import json
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from collections import Counter, defaultdict
from tqdm import tqdm
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from functools import partial

##############################
# 1. 세그멘테이션용 유틸 함수
##############################

# 세그멘테이션 결과에서 가장 많이 등장하는 색상을 추출
def get_dominant_color(image_array, mask):
    valid_pixels = image_array[mask > 0]
    if len(valid_pixels) == 0:
        return "#FFFFFF"
    pixels = valid_pixels.reshape(-1, 3)
    counts = Counter(map(tuple, pixels))
    dominant_color = max(counts.items(), key=lambda x: x[1])[0]
    return '#{:02x}{:02x}{:02x}'.format(*dominant_color)

# 전체 세그멘테이션용 라벨 매핑 (test.py에서 참조)
label_map = {
    0: "background",
    1: "hat",
    2: "hair",
    3: "sunglasses",
    4: "upper-clothes",
    5: "skirt",
    6: "pants",
    7: "dress",
    8: "belt",
    9: "left-shoe",
    10: "right-shoe",
    11: "face",
    12: "left-leg",
    13: "right-leg",
    14: "left-arm",
    15: "right-arm",
    16: "bag",
    17: "scarf"
}

# 부위별 그룹
upper_body_labels = {
    1: "hat",
    3: "sunglasses",
    4: "upper-clothes",
    11: "face",
    14: "left-arm",
    15: "right-arm",
}
middle_body_labels = {
    5: "skirt",
    6: "pants",
    7: "dress",
    8: "belt",
}
lower_body_labels = {
    9: "left-shoe",
    10: "right-shoe",
    12: "left-leg",
    13: "right-leg",
}

# 세그멘테이션 모델 & 프로세서 초기화
processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
softmax = nn.Softmax(dim=1)

# GPU 설정 추가
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def segment_image(image: Image.Image):
    """
    GPU를 사용하도록 수정된 세그멘테이션 함수
    """
    image_array = np.array(image)
    inputs = processor(images=image, return_tensors="pt")
    # GPU로 입력 데이터 이동
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # CPU로 결과 이동 후 처리
    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False
    ).cpu()
    
    pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
    conf = softmax(upsampled_logits)[0]
    confidence_scores = {
        label_map[i]: float(conf[i].max())
        for i in np.unique(pred_seg)
        if i != 0
    }
    return pred_seg, confidence_scores

def process_group(image_array, pred_seg, confidence_scores, group_labels):
    """
    특정 그룹(upper/middle/lower)을 마스킹하여
    결과 이미지, 그리고 각 라벨별 정보(색상, confidence 등)를 반환
    """
    # 배경을 흰색(255)으로 초기화
    result_image = np.ones_like(image_array) * 255
    group_data = {}
    for label_id, label_name in group_labels.items():
        mask = (pred_seg == label_id)
        if mask.any():
            # 마스킹된 부분만 원본 이미지를 복사
            result_image[mask] = image_array[mask]
            dominant_color = get_dominant_color(image_array, mask)
            # confidence가 존재하면 추가
            conf_val = confidence_scores.get(label_name, 0.0)
            group_data[label_name] = {
                "confidence": float(conf_val),
                "dominant_color": dominant_color
            }
    return result_image, group_data

########################
# 2. 추가: label -> bodyPath, label -> wearingInfoKey 매핑
########################
label_to_body_path = {
    # upper
    "hat": "upper",
    "sunglasses": "upper",
    "upper-clothes": "upper",
    "face": "upper",
    "left-arm": "upper",
    "right-arm": "upper",
    # middle
    "skirt": "middle",
    "pants": "middle",
    "dress": "middle",
    "belt": "middle",
    # lower
    "left-shoe": "lower",
    "right-shoe": "lower",
    "left-leg": "lower",
    "right-leg": "lower"
}

# wearing_info 상의 key와 세그멘테이션 label 간 매핑 예시
# 사용자가 원하는 키값들에 맞춰 적절히 조정
label_to_wearing_key = {
    "hat": "hat",
    "sunglasses": None,   # wearing_info에 sunglasses가 없다면 None
    "upper-clothes": "main_top",  # main_top / inner_top 등 논리에 따라 조정
    "skirt": "bottom",
    "pants": "bottom",
    "dress": "bottom",
    "belt": None,
    "left-shoe": "shoes",
    "right-shoe": "shoes",
    "face": None,
    "left-arm": None,
    "right-arm": None,
    "left-leg": None,
    "right-leg": None
}

def process_single_image(info_item, wearing_images_dir, seg_dir):
    """
    단일 이미지 처리를 위한 워커 함수
    """
    wearing_image_name = info_item["wearing"]
    wearing_image_path = os.path.join(wearing_images_dir, wearing_image_name)
    
    if not os.path.exists(wearing_image_path):
        return None

    ##################
    # 1) orientation 체크
    ##################
    with Image.open(wearing_image_path) as im_check:
        width, height = im_check.size
    # 가로가 더 길면(landscape) 해당 이미지는 처리 건너뜀
    if width > height:
        return None

    # 여기서부터 이미지 열어서 세그멘테이션
    image = Image.open(wearing_image_path).convert("RGB")
    pred_seg, confidence_scores = segment_image(image)
    image_array = np.array(image)

    # 부위별 segmentation 처리
    upper_result, upper_data = process_group(image_array, pred_seg, confidence_scores, upper_body_labels)
    middle_result, middle_data = process_group(image_array, pred_seg, confidence_scores, middle_body_labels)
    lower_result, lower_data = process_group(image_array, pred_seg, confidence_scores, lower_body_labels)

    # 결과 저장
    upper_path = os.path.join(seg_dir, f"{wearing_image_name}_upper.png")
    middle_path = os.path.join(seg_dir, f"{wearing_image_name}_middle.png")
    lower_path = os.path.join(seg_dir, f"{wearing_image_name}_lower.png")

    Image.fromarray(upper_result.astype('uint8')).save(upper_path)
    Image.fromarray(middle_result.astype('uint8')).save(middle_path)
    Image.fromarray(lower_result.astype('uint8')).save(lower_path)

    # 세그멘테이션 결과를 합침
    segmentation_info = {
        "upper_body": upper_data,
        "middle_body": middle_data,
        "lower_body": lower_data
    }

    ########################
    # 2) JSON 작성을 위한 데이터 구성
    ########################
    # item_data: {item_code: [ {product_image, wearing_image, category, color}, ... ], ...}
    item_data = defaultdict(list)

    # 세그멘테이션 label을 일괄적으로 조회
    all_segmentation_labels = {}
    all_segmentation_labels.update(upper_data)
    all_segmentation_labels.update(middle_data)
    all_segmentation_labels.update(lower_data)

    # 착용정보에 기재된 각 field와 매핑된 label이 있는지 확인
    # for 예: "hat": "1008013", "main_top": "1008011", "bottom": "1008012" ...
    #        세그멘테이션엔 "hat" -> hat, "upper-clothes" -> main_top ...
    for seg_label, seg_info in all_segmentation_labels.items():
        # seg_label: 예) "hat", "upper-clothes", ...
        wearing_key = label_to_wearing_key.get(seg_label, None)
        if wearing_key is None:
            # wearing_info 에 매핑할 항목이 없으면 패스
            continue
        if wearing_key not in info_item or info_item[wearing_key] is None:
            # 예: wearing_info에 main_top이 null 인 경우 등
            continue

        # 착용된 item_code 예) "1008013"
        item_code = info_item[wearing_key]
        if not item_code:
            continue

        # product_image는 {item_code}_F.jpg 쓴다 가정 (원하는 방식대로 지정 가능)
        product_image_name = f"{item_code}_F.jpg"

        # 해당 label이 upper/middle/lower 어디에 속하는지 보고, 그에 맞게 segmented 이미지 경로 선택
        body_part = label_to_body_path.get(seg_label, "upper")  # 기본값은 upper
        if body_part == "upper":
            seg_image_path = upper_path
        elif body_part == "middle":
            seg_image_path = middle_path
        elif body_part == "lower":
            seg_image_path = lower_path
        else:
            seg_image_path = upper_path

        # color (dominant_color)
        color_hex = seg_info.get("dominant_color", "#FFFFFF")

        # category: 세그멘테이션 모델이 뽑은 카테고리 (seg_label)
        category_label = seg_label

        # 하나의 객체 구성
        item_data[item_code].append({
            "product_image": product_image_name,
            "wearing_image": seg_image_path, 
            "category": category_label,
            "color": color_hex
        })

    # 최종 리턴: 세그 이미지를 저장한 상대경로와, 세그 정보, item_data
    return {
        "wearing_image": wearing_image_name,
        "image_seg_paths": {
            "upper": upper_path,
            "middle": middle_path,
            "lower": lower_path
        },
        "wearing_info": info_item,
        "segmentation_info": segmentation_info,
        "item_data": dict(item_data)  # defaultdict -> 일반 dict 변환
    }

def process_dataset(dataset_type: str):
    """
    병렬 처리를 적용한 데이터셋 처리 함수
    """
    data_root = os.path.join("data", dataset_type)
    wearing_images_dir = os.path.join(data_root, "wearing_images")
    info_path = os.path.join(data_root, "wearing_info.json")

    if not os.path.exists(info_path):
        print(f"{info_path} 가 존재하지 않아 처리할 수 없습니다.")
        return

    output_dir = os.path.join("preprocessing", dataset_type)
    os.makedirs(output_dir, exist_ok=True)
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)
    seg_dir = os.path.join(output_dir, "segmented_images")
    os.makedirs(seg_dir, exist_ok=True)

    with open(info_path, "r", encoding="utf-8") as f:
        wearing_info_list = json.load(f)

    num_workers = 16
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        process_fn = partial(process_single_image, wearing_images_dir=wearing_images_dir, seg_dir=seg_dir)
        results = list(tqdm(
            executor.map(process_fn, wearing_info_list),
            total=len(wearing_info_list),
            desc=f"Processing {dataset_type} dataset",
            ncols=100
        ))

    # None 제거
    result_json_list = [r for r in results if r is not None]

    # ↓ 기본 세그멘테이션 결과(기존) 저장
    output_label_path = os.path.join(labels_dir, f"{dataset_type}_segmentation_labels.json")
    with open(output_label_path, "w", encoding="utf-8") as f:
        json.dump(result_json_list, f, indent=4, ensure_ascii=False)
    print(f"★ {dataset_type} 데이터 세그멘테이션 완료. 결과: {output_label_path}")

    ####################################
    # 3) item_code 별 JSON 데이터 저장
    ####################################
    # 모든 item_code에 대한 데이터(append)
    item_code_map = defaultdict(list)
    for res in result_json_list:
        # "item_data": { "1008013": [ {...}, {...} ], "1008011": [ {...} ], ... }
        if "item_data" not in res:
            continue
        for code, dict_arr in res["item_data"].items():
            item_code_map[code].extend(dict_arr)

    # 각 item_code마다 {code}.json 형태로 저장/append
    for code, info_arr in item_code_map.items():
        output_file = os.path.join(labels_dir, f"{code}.json")
        # 이미 존재한다면 기존 데이터 불러오기
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as rf:
                old_data = json.load(rf)
        else:
            old_data = []

        # 이어 붙이기
        old_data.extend(info_arr)

        # 중복 제거 등을 하고 싶으면 여기서 처리
        # (단순히 합치기만 하는 경우 이 단계는 생략)

        # 최종 저장
        with open(output_file, "w", encoding="utf-8") as wf:
            json.dump(old_data, wf, indent=4, ensure_ascii=False)

def main():
    for ds_type in ["train", "val"]:
        process_dataset(ds_type)

if __name__ == "__main__":
    main() 