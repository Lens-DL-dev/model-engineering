import os
import json
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ContrastiveFashionDataset(Dataset):
    """
    각 샘플은 착용 이미지(wearing)와 in-shop 상품(product) 이미지의 pair로 구성됩니다.
    SupCon 방식에서는 (wearing_img, product_img, product_code, color_group)을 반환하고,
    batch 내에서 product_code(정답)와 달리지만 color_group이 같은 상품끼리는
    '더 어려운(비슷한 색상) 음성 예시'가 됩니다.
    """
    def __init__(
        self,
        root_dir,
        metainfo_path,
        color_group_path=None,    # 추가: color_batch_group.json 경로
        is_train=True,
        image_size=224,
        n_mask_channels=0,
        mode="supcon",
        max_samples=-1
    ):
        super().__init__()
        self.root_dir = root_dir
        self.is_train = is_train
        self.image_size = image_size
        self.n_mask_channels = n_mask_channels
        self.mode = mode

        # ----------------------------------------------------
        # (1) color_batch_group.json 불러와서 product->group_id 매핑 테이블 생성
        # ----------------------------------------------------
        self.product_color_map = {}  # 예: { "1014964": 0, "1110819": 0, "1124568": 0, ... }
        if color_group_path is not None and os.path.exists(color_group_path):
            with open(color_group_path, 'r', encoding='utf-8') as f:
                color_data = json.load(f)
            # color_data는 { "Pants": [{ "group_id": ..., "center_color": ..., "product_ids": [...] }, ...],
            #                "Shoes": [...], ... } 형태라고 가정
            for category_name, group_list in color_data.items():
                for group_info in group_list:
                    group_id = group_info["group_id"]
                    product_ids = group_info["product_ids"]
                    for pid in product_ids:
                        # group_id만 저장해도 되고, (category, group_id)를 tuple로 저장해도 됨
                        self.product_color_map[pid] = group_id
        
        # ----------------------------------------------------
        # (2) transforms 설정
        # ----------------------------------------------------
        if self.is_train:
            # (참고) 필요하면 ColorJitter, RandomErasing 등 더 강한 증강을 추가
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
        
        # ----------------------------------------------------
        # (3) metainfo.json 로드
        # ----------------------------------------------------
        with open(metainfo_path, 'r', encoding='utf-8') as f:
            self.info_list = json.load(f)
        
        # ----------------------------------------------------
        # (4) wearing_img, product_img 파일 경로 및 라벨(product_code) 수집
        # ----------------------------------------------------
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
            if (not os.path.exists(wearing_path)) or (not os.path.exists(product_path)):
                continue
            
            # color_group_id 가져오기 (없으면 -1)
            color_group_id = self.product_color_map.get(product_code, -1)
            
            self.samples.append({
                "wearing_path": wearing_path,
                "product_path": product_path,
                "product_code": product_code,
                "color_group": color_group_id
            })
        
        # max_samples 제한
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
        color_group = sample["color_group"]

        # ----------------------------------------------------
        # (5) transform 적용
        # ----------------------------------------------------
        if self.is_train:
            wearing_img = self.wearing_transform(wearing_img)
            product_img = self.product_transform(product_img)
        else:
            wearing_img = self.transform_val(wearing_img)
            product_img = self.transform_val(product_img)
        
        # SupCon 모드에서 (wearing_img, product_img, product_code, color_group)까지 반환
        return wearing_img, product_img, product_code, color_group

    def get_all_color_groups(self):
        """한 번에 모든 색상 그룹 정보를 반환"""
        group_dict = {}
        for idx, (_, _, _, color_group) in enumerate(self.samples):
            if color_group not in group_dict:
                group_dict[color_group] = []
            group_dict[color_group].append(idx)
        return group_dict

def collate_fn_supcon(batch):
    """
    batch: list of tuples: (wearing_img, product_img, product_code, color_group)
    반환:
      wearing_imgs: (B, C, H, W)
      product_imgs: (B, C, H, W)
      product_codes: list of length B
      color_groups:  list of length B
    """
    wearing_imgs = []
    product_imgs = []
    product_codes = []
    color_groups = []
    for (w_img, p_img, p_code, c_grp) in batch:
        wearing_imgs.append(w_img)
        product_imgs.append(p_img)
        product_codes.append(p_code)
        color_groups.append(c_grp)
    
    wearing_imgs = torch.stack(wearing_imgs, dim=0)
    product_imgs = torch.stack(product_imgs, dim=0)
    
    return wearing_imgs, product_imgs, product_codes, color_groups
