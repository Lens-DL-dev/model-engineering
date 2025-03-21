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
        - 착용 이미지에는 강한 augmentation (RandomResizedCrop, HorizontalFlip 등)
        - 상품 이미지에는 Resize와 Normalize (약한 augmentation)
    - is_train=False 인 경우:
        간단한 Resize 및 Normalize transform을 사용합니다.
    
    mode 인자를 통해 (향후 다른 모드도 지원 가능하지만, 여기서는 'supcon'만 사용)
    """
    def __init__(self, root_dir, metainfo_path, is_train=True, image_size=224, 
                 n_mask_channels=0, mode="supcon", max_samples=-1, config=None):
        super().__init__()
        self.root_dir = root_dir
        self.is_train = is_train
        self.image_size = image_size
        self.n_mask_channels = n_mask_channels
        self.mode = mode
        
        # config로부터 augmentation 강도 설정
        jitter_strength = 0.2
        min_scale = 0.9  # 기본값을 더 높게 설정 (0.7 -> 0.9)
        use_random_resize_crop = True
        crop_ratio_range = [0.95, 1.05]  # 기본 비율 범위 좁게 설정
        
        if config and 'augmentation' in config:
            jitter_strength = config['augmentation'].get('color_jitter_strength', 0.4)
            min_scale = config['augmentation'].get('min_scale', 0.9)
            use_random_resize_crop = config['augmentation'].get('use_random_resize_crop', True)
            crop_ratio_range = config['augmentation'].get('crop_ratio_range', [0.95, 1.05])

        if self.is_train:
            wearing_transforms = []
            if use_random_resize_crop:
                # 더 안전한 크롭 설정 - 원본에 더 가깝게 유지
                wearing_transforms.append(
                    transforms.RandomResizedCrop(
                        image_size, 
                        scale=(min_scale, 1.0),  # 최소 90% 이상의 원본 영역 유지
                        ratio=crop_ratio_range   # 거의 원본 비율 유지
                    )
                )
            else:
                # 크롭 대신 안전한 대안: 리사이즈 후 약간의 패딩과 랜덤 이동
                wearing_transforms.extend([
                    transforms.Resize((int(image_size * 0.95), int(image_size * 0.95))),
                    transforms.Pad(padding=int(image_size * 0.05)),
                    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05))
                ])
            
            # 색상 변환은 그대로 유지
            wearing_transforms.extend([
                transforms.RandomHorizontalFlip(),
                # transforms.RandomApply([
                #     transforms.ColorJitter(
                #         brightness=jitter_strength, 
                #         contrast=jitter_strength, 
                #         saturation=jitter_strength, 
                #         hue=jitter_strength/4)
                # ], p=0.8),
                # transforms.RandomGrayscale(p=0.2),
                # transforms.GaussianBlur(kernel_size=int(0.1 * image_size) // 2 * 2 + 1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])
            
            self.wearing_transform = transforms.Compose(wearing_transforms)
            
            # 상품 이미지는 augmentation 강도 낮게 유지
            self.product_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            # 검증 시에는 동일하게 유지
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
            
            # 새로운 카테고리 우선순위 리스트
            product_categories = [
                "Hat", "Sunglasses", "Upper-clothes", "Skirt", 
                "Pants", "Dress", "Shoes", "Bag", "Scarf"
            ]
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
