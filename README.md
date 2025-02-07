# Lens-Dev 👁️

> 일반 영상정보(이미지)에 상품 정보를 찾는 로직

## Pipeline 설명

-   **[1차 처리] 이미지 임베딩**

    -   openai/clip-vit-base-patch32 기반 모델로 착용 이미지를 임베딩화

-   **[2차 처리] 벡터화 및 검색**
    -   FAISS를 활용하여 미리 벡터화된 판매 이미지 DB에서 가장 유사한 제품을 검색
    -   FAISS 관련 함수는 `build_faiss_index.py` 파일 참고
    -   벡터 DB 생성 및 검색 관련 함수는 `build_faiss_index.py` 파일 참고

## 작업 수행 방식 (Train / Inference)

1. 환경 마련

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. 데이터셋 준비

> [!IMPORTANT]
>
> ### 사전 확인
>
> `data` 폴더 내에 "구매 이미지" 는 product_images 폴더에, "착용 이미지" 는 wearing_images 폴더에 존재해야 함

3. 모델 학습

```bash
python train.py --config config.yaml
```

4. FAISS 인덱스 생성
    > 여기서부터 일부 수정됨 / 추가 필요

## 폴더 구조

```bash
/
├── data/
│   ├── example/
│   │   └── brands.json
│   ├── train/
│   │   ├── meta_info/
│   │   ├── product_images/
│   │   ├── segmented_images/
│   │   ├── product_category.json
│   │   └── wearing_info.json
│   ├── val/
│   │   ├── meta_info/
│   │   ├── product_images/
│   │   ├── segmented_images/
│   │   └── wearing_info.json
│   └── ...
├── faiss_index/
│   ├── product.faiss          # (인덱스 저장)
│   ├── embeddings.npy         # (product 임베딩 전체)
│   ├── product_ids.npy        # (productId 리스트)
│   ├── brand_map.json         # (productId->brand 매핑)
│   ├── category_map.json      # (productId->category 매핑)
│   └── ...
├── models/
│   └── efficientnet_v2.py
├── utils/
│   ├── dataset.py
│   ├── losses.py
│   ├── metrics.py
│   ├── config.py
│   └── visualization.py       # (시각화 로직)
├── build_faiss_index.py       # (새) 인덱스 구축
├── search_faiss_index.py      # (새) 검색 수행
├── config.yaml
├── train.py
├── inference.py
└── requirements.txt
```


## `wearing_info.json` 파일 형식

```json
[
    {
        "wearing": "1008_A001_000.jpg",
        "hat": "1008013",
        "main_top": "1008011",
        "inner_top": null,
        "bottom": "1008012",
        "shoes": null
    },
    {
        "wearing": "1008_A001_044.jpg",
        "hat": "1008013",
        "main_top": "1008011",
        "inner_top": null,
        "bottom": "1008012",
        "shoes": null
    }
    ...
]
```

## 전처리 데이터 설명 📁

> 학습에 사용될 전처리된 데이터셋 구조 및 형식 설명

### 폴더 구조

```bash
/preprocessing
├── train/
│   ├── labels/                  # 세그멘테이션 결과 JSON
│   │   ├── 1008001.json        # 상품코드별 세그멘테이션 정보
│   │   ├── ...
│   │   └── train_segmentation_labels.json  # 전체 세그멘테이션 결과
│   │
│   │
│   ├── product_images/
│   │   ├── 1008001/
│   │   │   ├── 1008001_B.jpg
│   │   │   └── 1008001_F.jpg
│   │   └── ...
│   │
│   ├── segmented_images/        # 세그멘테이션된 이미지
│   │   ├── 1008_B035_132.jpg_upper.png
│   │   ├── 1008_B035_132.jpg_middle.png
│   │   ├── 1008_B035_132.jpg_lower.png
│   │   └── ...
│   │
│   └── missing_data.json        # 세그멘테이션 결측 정보
│
└── val/
    ├── labels/
│   ├── product_images/
    ├── segmented_images/
    └── missing_data.json
```

### JSON 파일 형식

#### 전체 세그멘테이션 결과 (`{dataset_type}_segmentation_labels.json`)

```json
[
    {
        "wearing_image": "1209_D001_000.jpg",
        "image_seg_paths": {
            "upper": "preprocessing/val/segmented_images/1209_D001_000.jpg_upper.png",
            "middle": "preprocessing/val/segmented_images/1209_D001_000.jpg_middle.png",
            "lower": "preprocessing/val/segmented_images/1209_D001_000.jpg_lower.png"
        },
        "wearing_info": {
            "wearing": "1209_D001_000.jpg",
            "hat": null,
            "main_top": "1209183",
            "inner_top": null,
            "bottom": null,
            "shoes": "1208231"
        },
        "segmentation_info": {
            "upper_body": {
                "upper-clothes": {
                    // 이 부분이 key에 해당
                    "confidence": 0.869,
                    "dominant_color": "#4b4a4f"
                }
            },
            "middle_body": {
                "dress": {
                    "confidence": 0.97,
                    "dominant_color": "#444348"
                }
            },
            "lower_body": {
                "left-shoe": {
                    "confidence": 0.995,
                    "dominant_color": "#0d0c11"
                }
            }
        }
    }
]
```

-   `wearing_image`: 원본 착용 이미지 파일명
-   `image_seg_paths`: 신체 부위별 세그멘테이션 이미지 경로
-   `wearing_info`: 원본 착용 정보 (상품 코드)
-   `segmentation_info`: 신체 부위별 세그멘테이션 결과
    -   (key): 신체 부위별로 구성됨
        -   (key): 세그멘테이션 모델이 분류한 카테고리
            -   `confidence`: 세그멘테이션 신뢰도 (0-1)
            -   `dominant_color`: 해당 부위의 주요 색상

#### 3. 결측 데이터 정보 (`missing_data.json`)

```json
{
    "1209_D001_000.jpg": [
        {
            "body_part": "middle_body",
            "parts": ["dress", "pants"],
            "num_parts": 2
        }
    ]
}
```

-   특정 신체 부위가 전체적으로 흰색(`#FFFFFF`)으로 세그멘테이션된 경우를 표시
-   `body_part`: 문제가 발생한 신체 부위 (`upper_body`, `middle_body`, `lower_body`)
-   `parts`: 해당 부위에 포함된 세부 파트 목록
-   `num_parts`: 세부 파트 개수

### 세그멘테이션 라벨 정보

```python
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
```
