# utils/dataset.py
import os
import json
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ContrastiveFashionDataset(Dataset):
    """
    하나의 anchor(착용이미지)에 대해
     - Positive(정답 in-shop) 1장
     - Negative(같은 category, 다른 product code) N장
    을 함께 반환
    """
    def __init__(
        self,
        root_dir,           
        metainfo_path,
        is_train=True,
        transform=None,
        image_size=224,
        n_mask_channels=1, # 250121_wsj 3(RGB) + Mask channel(0 ~ N)
        negative_count=4    # 사용자 설정: 한 anchor당 negative 몇 장?
    ):
        super().__init__()
        self.root_dir = root_dir
        self.is_train = is_train
        self.negative_count = negative_count
        self.n_mask_channels = n_mask_channels

        if transform is not None:
            self.transform = transform
        else:
            # wearing 이미지와 product 이미지에 대한 transform을 따로 정의
            self.product_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((image_size, image_size)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406] + [0] * n_mask_channels,
                    std=[0.229, 0.224, 0.225] + [1] * n_mask_channels
                )
            ])
            
            # wearing 이미지용 transform (augmentation 추가)
            self.wearing_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomResizedCrop(
                    size=image_size,
                    scale=(0.8, 1.0),  # 원본 크기의 80~100%
                    ratio=(0.9, 1.1)   # 가로세로 비율 유지
                ),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406] + [0] * n_mask_channels,
                    std=[0.229, 0.224, 0.225] + [1] * n_mask_channels
                )
            ])
            self.transform = None  # 기존 transform 변수는 None으로 설정

        # 1) wearing_info.json 로드
        with open(metainfo_path, 'r', encoding='utf-8') as f:
            self.wearing_list = json.load(f)

        # 2) anchor_items: (img_path, mask_path, product_code, category)
        self.anchor_items = []
        for info in self.wearing_list:
            wearing_filename = info["wearing"]
            base_name = os.path.splitext(wearing_filename)[0]
            img_path = os.path.join(root_dir, "wearing", wearing_filename)
            
            # 부위별 코드
            hat_code = info.get("hat", None)
            main_top_code = info.get("main_top", None)
            inner_top_code = info.get("inner_top", None)
            bottom_code = info.get("bottom", None)
            shoes_code = info.get("shoes", None)

            # hat
            if hat_code:
                mask_path = os.path.join(root_dir, "masks", "wearing", f"{base_name}_hat.png") if self.n_mask_channels > 0 else None
                self.anchor_items.append({
                    "img_path": img_path,
                    "mask_path": mask_path,
                    "product_code": hat_code,
                    "category": "hat"
                })
                    
            # main_top / inner_top (main_top 우선)
            used_top_code = main_top_code if main_top_code else inner_top_code
            used_top_class = "main_top" if main_top_code else ("inner_top" if inner_top_code else None)
            if used_top_code and used_top_class:
                mask_path = os.path.join(root_dir, "masks", "wearing", f"{base_name}_{used_top_class}.png") if self.n_mask_channels > 0 else None
                self.anchor_items.append({
                    "img_path": img_path,
                    "mask_path": mask_path,
                    "product_code": used_top_code,
                    "category": used_top_class
                })
            # bottom
            if bottom_code:
                mask_path = os.path.join(root_dir, "masks", "wearing", f"{base_name}_bottom.png") if self.n_mask_channels > 0 else None
                self.anchor_items.append({
                    "img_path": img_path,
                    "mask_path": mask_path,
                    "product_code": bottom_code,
                    "category": "bottom"
                })
            # shoes
            if shoes_code:
                mask_path = os.path.join(root_dir, "masks", "wearing", f"{base_name}_shoes.png") if self.n_mask_channels > 0 else None
                self.anchor_items.append({
                    "img_path": img_path,
                    "mask_path": mask_path,
                    "product_code": shoes_code,
                    "category": "shoes"
                })

        # 3) in-shop 이미지 구조화
        self.product_dict = {}  # product_dict[code] = [img_path, ...]
        self.category_map = {}  # category_map[cat][code] = [img_path, ...] (카테고리별 분리)

        product_root = os.path.join(root_dir, "product")
        for fname in os.listdir(product_root):
            #sub_path = os.path.join(product_root, code_folder)
            #if os.path.isdir(sub_path):
            #    imgs = []
            #    for fname in os.listdir(sub_path):
            #        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            #            full_path = os.path.join(sub_path, fname)
            #            imgs.append(full_path)
            #    if imgs:
            #        self.product_dict[code_folder] = imgs
            
            # product 이미지가 front, back 2장에서 단일로 바뀜에 따라 수정
            
            imgs = []
            product_code = os.path.splitext(fname)[0]
            full_path = os.path.join(product_root, fname)
            imgs.append(full_path)
            self.product_dict[product_code] = imgs

        # 카테고리별로 정리하기 위해 anchor_items에 나타난 cat, code에 대한 mapping
        # (단, 실제 product에도 code가 있어야 의미가 있음)
        for item in self.anchor_items:
            cat = item["category"]
            code = item["product_code"]
            if code not in self.product_dict:
                continue
            if cat not in self.category_map:
                self.category_map[cat] = {}
            if code not in self.category_map[cat]:
                self.category_map[cat][code] = self.product_dict[code]

    def __len__(self):
        return len(self.anchor_items)

    def __getitem__(self, index):
        """
        반환:
          anchor_img, 
          [pos_img, neg_1, neg_2, ..., neg_n],
          [1, 0, 0, ..., 0]
        """
        anchor_info = self.anchor_items[index]
        anchor_img_path = anchor_info["img_path"]
        anchor_mask_path = anchor_info["mask_path"]
        anchor_code = anchor_info["product_code"]
        category = anchor_info["category"]

        # anchor 이미지 로드
        anchor_img = self._load_image_with_mask(anchor_img_path, anchor_mask_path)  #250120_kdi ; 기존 : self._load_image(anchor_path)
        if anchor_img is None:
            return self.__getitem__(random.randint(0, len(self)-1))
        
        # Positive 이미지 1장 (anchor_code의 in-shop 중 하나)
        pos_img = self._get_positive_item(anchor_code)
        if pos_img is None:
            return self.__getitem__(random.randint(0, len(self)-1))
            
        # Negative 이미지 N장 (같은 category, 다른 code)
        neg_imgs = self._get_negative_item(anchor_code, category)
        if neg_imgs is None:
            return self.__getitem__(random.randint(0, len(self)-1))
        
        # 최종 candidate 리스트 & 레이블
        candidate_list = [pos_img] + neg_imgs  # 길이: 1 + N
        label_list = [1] + [0]*self.negative_count

        # transform 적용 (wearing과 product 이미지 구분)
        anchor_img = self.wearing_transform(anchor_img)
        for i in range(len(candidate_list)):
            candidate_list[i] = self.product_transform(candidate_list[i])

        return anchor_img, candidate_list, torch.tensor(label_list, dtype=torch.float)

    def _get_positive_item(self, anchor_code):
        if anchor_code in self.product_dict:
            pos_img_path = random.choice(self.product_dict[anchor_code])
            if self.n_mask_channels == 0:
                pos_mask_path = None
            elif self.n_mask_channels == 1:
                pos_mask_path = os.path.join(self.root_dir, "masks", "product", os.path.basename(pos_img_path))
            else:
                raise NotImplementedError
            
            return self._load_image_with_mask(pos_img_path, pos_mask_path)
        else: 
            return None # anchor_code에 해당하는 in-shop이 없으면, 사실상 Positive가 없음 -> fallback
        
    def _get_negative_item(self, anchor_code, category):
        neg_imgs = []
        if (category in self.category_map) and (len(self.category_map[category]) > 1):
            # 현재 code가 아닌 다른 code들
            diff_codes = [c for c in self.category_map[category].keys() if c != anchor_code]
            if not diff_codes:
                # 다른 code가 없으면 fallback
                return None

            for _ in range(self.negative_count):
                neg_code = random.choice(diff_codes)
                neg_image_path = random.choice(self.category_map[category][neg_code])
                
                if self.n_mask_channels == 0:
                    neg_mask_path = None
                elif self.n_mask_channels == 1:
                    neg_mask_path = os.path.join(self.root_dir, "masks", "product", os.path.basename(neg_image_path))
                else:
                    raise NotImplementedError
                
                neg_img = self._load_image_with_mask(neg_image_path, neg_mask_path) #250120_kdi 기존 ;_load_image(neg_path)
                
                if neg_img is None:
                    return None
                neg_imgs.append(neg_img)
                
            return neg_imgs
        else:
            # category_map[cat]이 1개 code뿐이거나, cat 자체가 없는 경우
            return None
        
    def _load_image(self, path):
        img = Image.open(path).convert('RGB')
        ## 가로가 세로보다 긴 이미지면 None
        #if img.width > img.height:
        #    return None
        return img
    
    def _load_image_with_mask(self, img_path, mask_path): # 250120_kdi 추가
            img = Image.open(img_path).convert('RGB')
            #if img.width > img.height:
            #    return None
        
            # mask 데이터를 사용하지 않는 경우 img만 리턴
            if mask_path is None:
                return img
            
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L')
            else:
                #raise FileNotFoundError(f"Mask image for {img_path} not found")
                mask = Image.new('L', img.size, color=0)  # 빈 마스크 생성

            img = np.array(img)
            mask = np.array(mask)[:, :, np.newaxis] # 차원 추가 [H, W, 1]
            combined = np.concatenate((img, mask.astype(np.uint8)), axis=-1)  # [H, W, 4] 
            return combined

def collate_fn_contrastive(batch):
    """
    batch: list of tuples:
      (anchor_img, [cand_img1, cand_img2, ...], [label1, label2, ...])
    각 anchor에 대해 (N+1)개의 candidate가 있음.
    => 이를 하나의 큰 텐서로 펼쳐서 반환
       anchors : (batch*(N+1), C, H, W)
       cands   : (batch*(N+1), C, H, W)
       labels  : (batch*(N+1))
    """
    anchor_list = []
    candidate_list = []
    label_list = []

    for (anchor_img, cand_imgs, labels) in batch:
        # anchor_img shape: (C,H,W)
        # cand_imgs shape: list of length (N+1)
        # labels shape: (N+1,)

        # anchor를 (N+1)번 반복
        for i in range(len(cand_imgs)):
            anchor_list.append(anchor_img)  # 동일 anchor
            candidate_list.append(cand_imgs[i])
            label_list.append(labels[i])

    anchors = torch.stack(anchor_list, dim=0)
    candidates = torch.stack(candidate_list, dim=0)
    labels = torch.stack(label_list, dim=0)

    return anchors, candidates, labels