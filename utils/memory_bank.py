import torch

class MemoryBank:
    """
    메모리 뱅크: 최근 샘플들의 임베딩을 저장하여 negative 샘플 풀을 확장합니다
    A100 80GB GPU에 최적화되어 더 큰 메모리 뱅크 크기 지원
    """
    def __init__(self, size=16384, dim=512, device="cuda", momentum=0.99):
        self.size = size
        self.dim = dim
        self.device = device
        self.bank = torch.randn(size, dim).to(device)
        self.bank = torch.nn.functional.normalize(self.bank, dim=1)
        self.product_codes = ["" for _ in range(size)]
        self.ptr = 0
        self.is_full = False
        self.momentum = momentum  # 모멘텀 업데이트 계수 추가

    def get(self):
        """현재 메모리 뱅크에 저장된 임베딩과 코드를 반환합니다"""
        if self.is_full:
            return self.bank, self.product_codes
        else:
            return self.bank[:self.ptr], self.product_codes[:self.ptr]
    
    def get_size(self):
        """현재 메모리 뱅크에 저장된 샘플 수를 반환합니다"""
        return self.size if self.is_full else self.ptr

    def enqueue_and_dequeue(self, embs, codes):
        """
        새로운 임베딩을 메모리 뱅크에 추가합니다
        모멘텀 업데이트 사용 시, 기존 임베딩과 새 임베딩을 블렌딩
        """
        batch_size = embs.shape[0]
        
        # 포인터 위치부터 배치 크기만큼의 위치에 새 임베딩 저장
        if self.ptr + batch_size > self.size:
            # 메모리 뱅크의 끝에 도달한 경우
            remaining = self.size - self.ptr
            
            # 모멘텀 업데이트 적용 (첫 부분)
            if self.is_full and self.momentum > 0:
                # 기존 임베딩과 새 임베딩을 블렌딩
                self.bank[self.ptr:] = self.momentum * self.bank[self.ptr:] + (1 - self.momentum) * embs[:remaining].detach()
                self.bank[self.ptr:] = torch.nn.functional.normalize(self.bank[self.ptr:], dim=1)
                
                # 두 번째 부분도 모멘텀 업데이트
                self.bank[:batch_size-remaining] = self.momentum * self.bank[:batch_size-remaining] + (1 - self.momentum) * embs[remaining:].detach()
                self.bank[:batch_size-remaining] = torch.nn.functional.normalize(self.bank[:batch_size-remaining], dim=1)
            else:
                # 첫 채우기 시 모멘텀 없이 직접 할당
                self.bank[self.ptr:] = embs[:remaining].detach()
                self.bank[:batch_size-remaining] = embs[remaining:].detach()
            
            # 상품 코드도 같은 방식으로 업데이트
            self.product_codes[self.ptr:] = codes[:remaining]
            self.product_codes[:batch_size-remaining] = codes[remaining:]
            
            self.ptr = (self.ptr + batch_size) % self.size
            self.is_full = True
        else:
            # 메모리 뱅크에 공간이 충분한 경우
            if self.is_full and self.momentum > 0:
                # 모멘텀 업데이트 적용
                self.bank[self.ptr:self.ptr+batch_size] = self.momentum * self.bank[self.ptr:self.ptr+batch_size] + (1 - self.momentum) * embs.detach()
                self.bank[self.ptr:self.ptr+batch_size] = torch.nn.functional.normalize(self.bank[self.ptr:self.ptr+batch_size], dim=1)
            else:
                # 첫 채우기 시 모멘텀 없이 직접 할당
                self.bank[self.ptr:self.ptr+batch_size] = embs.detach()
                
            self.product_codes[self.ptr:self.ptr+batch_size] = codes
            self.ptr = (self.ptr + batch_size) % self.size
            if self.ptr == 0:
                self.is_full = True
    
    def reset(self):
        """메모리 뱅크를 초기화합니다"""
        self.bank = torch.randn(self.size, self.dim).to(self.device)
        self.bank = torch.nn.functional.normalize(self.bank, dim=1)
        self.product_codes = ["" for _ in range(self.size)]
        self.ptr = 0
        self.is_full = False
    
    def to(self, device):
        """메모리 뱅크를 지정된 디바이스로 이동합니다"""
        self.device = device
        self.bank = self.bank.to(device)
        return self 