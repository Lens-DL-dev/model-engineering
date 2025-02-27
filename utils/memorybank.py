class MemoryBank:
    """
    간단한 FIFO 큐 형태 메모리 뱅크.
    - embeddings: (N, D) 저장
    - product_codes: list of length N
    - color_groups: list of length N
    """
    def __init__(self, max_size=1024, embedding_dim=1024, device='cuda'):
        self.max_size = max_size
        self.device = device

        self.embeddings = torch.empty(0, embedding_dim, device=self.device)
        self.product_codes = []
        self.color_groups = []

    def __len__(self):
        return len(self.product_codes)

    def push(self, new_emb, new_codes, new_colors):
        """
        new_emb: (B, D) tensor
        new_codes: list of length B
        new_colors: list of length B
        """
        # FIFO로 저장
        if self.embeddings.size(0) == 0:
            # 메모리가 비어있으면 바로 세팅
            self.embeddings = new_emb.clone().detach()
            self.product_codes = new_codes[:]
            self.color_groups = new_colors[:]
        else:
            # concat
            self.embeddings = torch.cat([self.embeddings, new_emb], dim=0)
            self.product_codes.extend(new_codes)
            self.color_groups.extend(new_colors)

        # 만약 max_size 초과하면 앞에서부터 제거
        current_size = self.embeddings.size(0)
        if current_size > self.max_size:
            overflow = current_size - self.max_size
            # 앞쪽 overflow 개수를 버림
            self.embeddings = self.embeddings[overflow:, :]
            self.product_codes = self.product_codes[overflow:]
            self.color_groups = self.color_groups[overflow:]

    def get_all(self):
        """
        전체를 반환
        Return:
          emb: (N, D) tensor
          codes: list of length N
          colors: list of length N
        """
        return self.embeddings, self.product_codes, self.color_groups