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
    SupCon 방식에서는 각 샘플마다 (wearing_img, product_img, product_code)를 반환합니다.
    
    - is_train=True 인 경우:
        - 착용 이미지에는 강한 augmentation (RandomResizedCrop, HorizontalFlip, ColorJitter, GaussianBlur, RandomErasing 등)
        - 상품 이미지에는 Resize와 Normalize (약한 augmentation)
    - is_train=False 인 경우:
        간단한 Resize 및 Normalize transform을 사용합니다.
    
    mode 인자를 통해 (향후 다른 모드도 지원 가능하지만, 여기서는 'supcon'만 사용)
    """
    def __init__(self, root_dir, metainfo_path, is_train=True, image_size=224, n_mask_channels=0, mode="supcon", max_samples=-1):
        super().__init__()
        self.root_dir = root_dir
        self.is_train = is_train
        self.image_size = image_size
        self.n_mask_channels = n_mask_channels
        self.mode = mode

        if self.is_train:
            self.wearing_transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            self.product_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform_val = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        
        # metainfo.json 파일 로드 (각 entry는 'wearing' 및 상품 관련 키(예: "hat", "main_top", "inner_top", "bottom", "shoes") 포함)
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
            product_code = None
            for cat in product_categories:
                if info.get(cat) is not None:
                    product_code = info.get(cat)
                    break
            if product_code is None:
                continue
            product_path = os.path.join(root_dir, "product", f"{product_code}.jpg")
            if not os.path.exists(wearing_path) or not os.path.exists(product_path):
                continue
            self.samples.append({
                "wearing_path": wearing_path,
                "product_path": product_path,
                "product_code": product_code
            })
        
        if max_samples > 0 and len(self.samples) > max_samples:
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]
            print(f"Using {max_samples} samples out of the full dataset")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample = self.samples[index]
        wearing_img = Image.open(sample["wearing_path"]).convert("RGB")
        product_img = Image.open(sample["product_path"]).convert("RGB")
        product_code = sample["product_code"]
        
        if self.is_train:
            wearing_img = self.wearing_transform(wearing_img)
            product_img = self.product_transform(product_img)
        else:
            wearing_img = self.transform_val(wearing_img)
            product_img = self.transform_val(product_img)
        
        # SupCon 모드: (wearing_img, product_img, product_code) 반환
        return wearing_img, product_img, product_code

def collate_fn_supcon(batch):
    """
    batch: list of tuples: (wearing_img, product_img, product_code)
    반환:
      wearing_imgs: (B, C, H, W)
      product_imgs: (B, C, H, W)
      product_codes: list of length B (각 샘플의 상품 코드, 문자열)
    """
    wearing_imgs = []
    product_imgs = []
    product_codes = []
    for wearing_img, product_img, prod_code in batch:
        wearing_imgs.append(wearing_img)
        product_imgs.append(product_img)
        product_codes.append(prod_code)
    wearing_imgs = torch.stack(wearing_imgs, dim=0)
    product_imgs = torch.stack(product_imgs, dim=0)
    return wearing_imgs, product_imgs, product_codes
