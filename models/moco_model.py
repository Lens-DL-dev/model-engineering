import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, embed_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.bn1 = nn.LayerNorm(in_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_dim, embed_dim)
        self.bn2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.normalize(x, dim=1)
        return x

def get_backbone(backbone_name="convnext_tiny", pretrained=True):
    if backbone_name == "convnext_tiny":
        encoder = models.convnext_tiny(pretrained=pretrained)
        in_dim = encoder.classifier[2].in_features
        encoder.classifier = nn.Identity()
    else:
        raise NotImplementedError(f"Backbone {backbone_name} not implemented")
    return encoder, in_dim

class MoCoModel(nn.Module):
    def __init__(self, backbone="convnext_tiny", pretrained=True, embed_dim=256, m=0.999, K=4096, temperature=0.07):
        super().__init__()
        self.K = K
        self.m = m
        self.temperature = temperature

        # Query encoder
        self.encoder_q, in_dim = get_backbone(backbone, pretrained)
        self.projection_head_q = ProjectionHead(in_dim, embed_dim)

        # Key encoder
        self.encoder_k, _ = get_backbone(backbone, pretrained)
        self.projection_head_k = ProjectionHead(in_dim, embed_dim)
        self._init_key_encoder()

        # Create the negative queue (size: embed_dim x K)
        self.register_buffer("queue", torch.randn(embed_dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    @torch.no_grad()
    def _init_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        for param_q, param_k in zip(self.projection_head_q.parameters(), self.projection_head_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.projection_head_q.parameters(), self.projection_head_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + batch_size > self.K:
            remaining = self.K - ptr
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue[:, :batch_size - remaining] = keys[remaining:].T
            self.queue_ptr[0] = batch_size - remaining
        else:
            self.queue[:, ptr:ptr+batch_size] = keys.T
            self.queue_ptr[0] = (ptr + batch_size) % self.K

    def forward(self, im_q, im_k):
        q = self.encoder_q(im_q)
        q = self.projection_head_q(q)
        q = F.normalize(q, dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k)
            k = self.projection_head_k(k)
            k = F.normalize(k, dim=1)
        
        # Compute logits
        # Positive logits: (B, 1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # Negative logits: (B, K)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.temperature

        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        self._dequeue_and_enqueue(k)
        return logits, labels
