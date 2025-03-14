import torch

class MemoryBank:
    """
    메모리 뱅크: 최근 샘플들의 임베딩을 저장하여 negative 샘플 풀을 확장합니다
    """
    def __init__(self, size=4096, dim=512, device="cuda"):
        self.size = size
        self.dim = dim
        self.device = device
        self.bank = torch.randn(size, dim).to(device)
        self.bank = torch.nn.functional.normalize(self.bank, dim=1)
        self.product_codes = ["" for _ in range(size)]
        self.ptr = 0
        self.is_full = False

    def get(self):
        """현재 메모리 뱅크에 저장된 임베딩과 코드를 반환합니다"""
        if self.is_full:
            return self.bank, self.product_codes
        else:
            return self.bank[:self.ptr], self.product_codes[:self.ptr]

    def enqueue_and_dequeue(self, embs, codes):
        """새로운 임베딩을 메모리 뱅크에 추가합니다"""
        batch_size = embs.shape[0]
        
        # 포인터 위치부터 배치 크기만큼의 위치에 새 임베딩 저장
        if self.ptr + batch_size > self.size:
            # 메모리 뱅크의 끝에 도달한 경우
            remaining = self.size - self.ptr
            self.bank[self.ptr:] = embs[:remaining].detach()
            self.bank[:batch_size-remaining] = embs[remaining:].detach()
            
            # 상품 코드도 같은 방식으로 업데이트
            self.product_codes[self.ptr:] = codes[:remaining]
            self.product_codes[:batch_size-remaining] = codes[remaining:]
            
            self.ptr = (self.ptr + batch_size) % self.size
            self.is_full = True
        else:
            # 메모리 뱅크에 공간이 충분한 경우
            self.bank[self.ptr:self.ptr+batch_size] = embs.detach()
            self.product_codes[self.ptr:self.ptr+batch_size] = codes
            self.ptr = (self.ptr + batch_size) % self.size
            if self.ptr == 0:
                self.is_full = True 