# Lens-Dev 👁️

> 일반 영상정보(이미지)에 상품 정보를 찾는 로직

## Pipeline 설명

-   **[1차 처리] 이미지 임베딩**

    -   ConvNeXT 기반 모델로 착용 이미지를 임베딩화

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

```bash
python ./faiss/build_faiss_index.py
```

5. 추론

## 폴더 구조

```bash
.
├── README.md
├── config.yaml
├── faiss
│   ├── build_faiss_index.py
│   └── search_faiss_index.py
├── models
│   └── convnext.py
├── requirements.txt
├── train.py
└── utils
    ├── config.py
    ├── dataset.py
    ├── losses.py
    ├── metrics.py
    └── visualization.py
```

## SimCLR 학습 방법

SimCLR(Simple Framework for Contrastive Learning of Representations)은 비지도 학습 방식으로, 레이블 정보 없이 contrastive learning을 수행합니다.

### SimCLR 학습 실행

```bash
python train_simclr.py --config config_simclr.yaml
```

### SimCLR vs SupCon

-   **SimCLR**: 각 이미지에 두 가지 다른 augmentation을 적용하여 positive pair 생성. 배치 내 다른 모든 샘플은 negative로 간주. 레이블 정보 사용하지 않음.
-   **SupCon**: 레이블 정보를 활용해 같은 클래스의 샘플들을 positive로 간주. 다른 클래스의 샘플들은 negative로 간주.
