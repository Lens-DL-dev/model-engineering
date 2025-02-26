import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ContrastiveFashionDataset(Dataset):
    """
    각 샘플은 착용 이미지와 in-shop 상품 이미지의 pair로 구성됩니다.
    - 착용 이미지 (wearing): 강한 augmentation (SimCLR/MoCo view)
    - 상품 이미지 (product): 상대적으로 약한 augmentation
    두 transform 모두 ImageNet 평균/표준편차로 정규화합니다.
    """
    def __init__(self, root_dir, metainfo_path, is_train=True, image_size=384, n_mask_channels=0, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.is_train = is_train
        self.image_size = image_size
        self.n_mask_channels = n_mask_channels

        if transform is not None:
            self.transform = transform
        else:
            # 강한 augmentation (착용 이미지)
            self.wearing_transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
                transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.33))
            ])
            # 상품 이미지에 대한 약한 augmentation:
            self.product_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        # metainfo.json 파일에는 각 샘플에 대해 "wearing"과 하나 이상의 상품 관련 키(예: "hat", "main_top", ...)가 포함됨
        with open(metainfo_path, 'r', encoding='utf-8') as f:
            self.info_list = json.load(f)

        self.samples = []
        for info in self.info_list:
            wearing_filename = info.get("wearing", None)
            if wearing_filename is None:
                continue
            wearing_path = os.path.join(root_dir, "wearing", wearing_filename)
            # 우선순위: hat, main_top, inner_top, bottom, shoes
            product_categories = ["hat", "main_top", "inner_top", "bottom", "shoes"]
            product_code = next((info.get(cat) for cat in product_categories if info.get(cat) is not None), None)
            if product_code is None:
                continue
            product_path = os.path.join(root_dir, "product", f"{product_code}.jpg")
            if not os.path.exists(wearing_path) or not os.path.exists(product_path):
                continue
            self.samples.append({
                "wearing_path": wearing_path,
                "product_path": product_path
            })

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample = self.samples[index]
        wearing_img = Image.open(sample["wearing_path"]).convert("RGB")
        product_img = Image.open(sample["product_path"]).convert("RGB")
        if self.is_train:
            wearing_img = self.wearing_transform(wearing_img)
            product_img = self.product_transform(product_img)
        else:
            # 검증 시 간단한 transform 적용
            transform_val = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            wearing_img = transform_val(wearing_img)
            product_img = transform_val(product_img)
        return wearing_img, product_img
