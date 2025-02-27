# sampler.py
import math
import random
from torch.utils.data import Sampler

class ColorGroupSampler(Sampler):
    """
    Dataset 내 color_group별로 인덱스를 모아두고,
    한 배치를 구성할 때 가급적 '동일 color_group' 샘플들만 모아서
    하드 네거티브(색상은 같지만 product_code가 다른) 비율을 높이려는 간단한 샘플러 예시.

    만약 한 color_group에 충분한 샘플이 없으면 다른 group도 섞어서 배치를 채움.
    """

    def __init__(self, dataset, batch_size, shuffle=True):
        """
        dataset: ContrastiveFashionDataset (wearing_img, product_img, product_code, color_group)
        batch_size: int
        shuffle: bool
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # color_group -> list of indices를 미리 캐시
        self.group_dict = {}
        
        # 데이터셋에서 한 번에 color_group 정보 가져오기
        if hasattr(dataset, 'get_all_color_groups'):
            self.group_dict = dataset.get_all_color_groups()
        else:
            # 기존 방식으로 fallback
            for idx in range(len(dataset)):
                _, _, _, color_group = dataset[idx]
                if color_group not in self.group_dict:
                    self.group_dict[color_group] = []
                self.group_dict[color_group].append(idx)

        self.all_groups = list(self.group_dict.keys())
        self.total_size = len(dataset)

    def __iter__(self):
        # epoch마다 group 순서를 섞고, group 내부 인덱스들도 섞음
        if self.shuffle:
            random.shuffle(self.all_groups)
            for g in self.all_groups:
                random.shuffle(self.group_dict[g])

        batch_buffer = []
        # group 단위로 순회하며 batch_size씩 뽑음
        for g in self.all_groups:
            idx_list = self.group_dict[g]
            i = 0
            while i < len(idx_list):
                batch_chunk = idx_list[i : i + self.batch_size]
                i += self.batch_size
                
                if len(batch_chunk) == self.batch_size:
                    # 딱 batch_size면 바로 yield
                    yield batch_chunk
                else:
                    # batch_size보다 작은 경우 버퍼에 추가
                    batch_buffer.extend(batch_chunk)
                    
                # 버퍼가 batch_size 이상이면 처리
                while len(batch_buffer) >= self.batch_size:
                    yield batch_buffer[:self.batch_size]
                    batch_buffer = batch_buffer[self.batch_size:]

        # leftover 처리
        if len(batch_buffer) > 0:
            if self.shuffle:
                random.shuffle(batch_buffer)
            while len(batch_buffer) >= self.batch_size:
                yield batch_buffer[:self.batch_size]
                batch_buffer = batch_buffer[self.batch_size:]
            # 마지막 불완전한 배치도 반환 (선택사항)
            if len(batch_buffer) > 0:
                yield batch_buffer

    def __len__(self):
        return math.ceil(self.total_size / self.batch_size)
