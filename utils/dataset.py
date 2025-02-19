# utils/dataset.py
import os
import json
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ContrastiveFashionDataset(Dataset):
    """
    각 샘플은 착용 이미지와 in-shop 상품 이미지의 pair로 구성됩니다.
    - 착용 이미지(wearing): 강한 augmentation (SimCLR/MoCo에서 view1)
    - 상품 이미지(product): 상대적으로 약한 augmentation (view2)
    """
    def __init__(self, root_dir, metainfo_path, is_train=True, image_size=384, n_mask_channels=0, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.is_train = is_train
        self.n_mask_channels = n_mask_channels
        self.image_size = image_size

        # 별도 transform 설정 (도메인 별 augmentation)
        if transform is not None:
            self.transform = transform
        else:
            self.wearing_transform = transforms.Compose([
                # transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406] + [0]*n_mask_channels,
                                     std=[0.229, 0.224, 0.225] + [1]*n_mask_channels)
            ])
            self.product_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406] + [0]*n_mask_channels,
                                     std=[0.229, 0.224, 0.225] + [1]*n_mask_channels)
            ])

        # metainfo.json 로드 (예: 각 항목에 "wearing"과 "product" 필드 포함)
        with open(metainfo_path, 'r', encoding='utf-8') as f:
            self.info_list = json.load(f)

        self.samples = []
        for info in self.info_list:
            wearing_filename = info["wearing"]
            wearing_path = os.path.join(root_dir, "wearing", wearing_filename)
            
            # Find the first non-null product code
            product_categories = ["hat", "main_top", "inner_top", "bottom", "shoes"]
            product_code = next((info[cat] for cat in product_categories if info[cat] is not None), None)
            
            if product_code is None or not os.path.exists(wearing_path):
                continue
                
            product_path = os.path.join(root_dir, "product", f"{product_code}.jpg")
            self.samples.append({
                "wearing_path": wearing_path,
                "product_path": product_path,
                "category": product_categories # 새로 추가함
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
            # 간단한 resize + normalize (validation)
            transform_val = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            wearing_img = transform_val(wearing_img)
            product_img = transform_val(product_img)
        return wearing_img, product_img
