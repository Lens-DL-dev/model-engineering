import os
import argparse
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torch.amp import GradScaler, autocast
from tqdm import tqdm
import wandb

from utils.config import load_config
from utils.dataset import ContrastiveFashionDataset
from utils.losses import NTXentLoss, InfoNCELoss
from utils.visualization import save_visualization, save_contrastive_matrix, save_topk_image_samples

# Import model classes
from models.simclr_model import SimCLRModel
from models.moco_model import MoCoModel

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def load_checkpoint(model, optimizer, filename):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print(f"=> Loaded checkpoint from '{filename}' (Epoch {start_epoch})")
        return start_epoch
    else:
        print(f"=> No checkpoint found at '{filename}'. Training from scratch.")
        return 0

def compute_topk_accuracy(query_emb, candidate_emb, topk=(1, 5, 10)):
    """
    query_emb, candidate_emb: (N, D) normalized embeddings.
    각 query의 정답은 동일 인덱스의 candidate라고 가정.
    """
    # (N, N) 유사도 행렬 (내적)
    similarity = torch.matmul(query_emb, candidate_emb.t())
    # 내림차순 정렬
    _, indices = similarity.topk(max(topk), dim=1)
    gt = torch.arange(query_emb.shape[0], device=query_emb.device).unsqueeze(1)
    correct = indices.eq(gt)
    topk_acc = {}
    for k in topk:
        topk_acc[k] = correct[:, :k].any(dim=1).float().mean().item() * 100.0
    return topk_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', type=str)
    args = parser.parse_args()
    
    
    # -----------------------------
    # 1. Load Config
    # -----------------------------

    config = load_config(args.config)
    use_wandb = config.get('use_wandb', False)
    if use_wandb:
        wandb.init(project="fashion-contrast", config=config)
        run_name = wandb.run.name
    else:
        run_name = datetime.now().strftime("%m%d%H%M%S")
    
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    lr = config['training']['learning_rate']
    num_workers = config['training']['num_workers']
    grad_acc_steps = config['training'].get('gradient_accumulation_steps', 1)
    early_stopping_patience = config['training'].get('early_stopping_patience', 3)
    
    # checkpoint
    save_dir = os.path.join(config['training'].get('save_dir', './checkpoints'), run_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # -----------------------------
    # 2. Device
    # -----------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # -----------------------------
    # 3. Dataset / Dataloader
    # -----------------------------

    train_dir = config['data']['train_dir']
    train_metainfo = config['data']['train_metainfo_path']
    val_dir = config['data']['val_dir']
    val_metainfo = config['data']['val_metainfo_path']
    
    image_size = config['model'].get('image_size', 384)
    n_mask_channels = config['data'].get('n_mask_channels', 0)
    # 데이터셋 크기 제한 옵션 추가
    max_samples = config['data'].get('max_samples', -1)  # -1은 전체 사용을 의미
    
    train_dataset = ContrastiveFashionDataset(root_dir=train_dir,
                                               metainfo_path=train_metainfo,
                                               is_train=True,
                                               image_size=image_size,
                                               n_mask_channels=n_mask_channels,
                                               max_samples=max_samples)
    val_dataset = ContrastiveFashionDataset(root_dir=val_dir,
                                             metainfo_path=val_metainfo,
                                             is_train=False,
                                             image_size=image_size,
                                             n_mask_channels=n_mask_channels,
                                             max_samples=max_samples)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, drop_last=False)
    
    vis_dir = config['training'].get('vis_dir', './vis_results')
    os.makedirs(vis_dir, exist_ok=True)
    
    # -----------------------------
    # 4. Model / Loss / Optimizer
    # -----------------------------
    method = config['model'].get('method', 'simclr').lower()
    embed_dim = config['model'].get('embed_dim', 256)
    backbone = config['model'].get('backbone', 'convnext_tiny')
    pretrained = config['model'].get('pretrained', True)
    
    if method == 'simclr':
        model = SimCLRModel(backbone=backbone, pretrained=pretrained, embed_dim=embed_dim)
        criterion = NTXentLoss(temperature=config['loss'].get('temperature', 0.07))
    elif method == 'moco':
        model = MoCoModel(backbone=backbone, pretrained=pretrained, embed_dim=embed_dim,
                          temperature=config['loss'].get('temperature', 0.07))
        criterion = InfoNCELoss()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler = GradScaler()
    
    # -----------------------------
    # 5. (Optional) Resume Checkpoint
    # -----------------------------
    checkpoint_path = config['model'].get('checkpoint', "")
    resume = config['model'].get('resume', False)
    start_epoch = 0
    if resume and checkpoint_path:
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
    
    best_val_top1 = 0.0
    epochs_no_improve = 0

    # 에폭당 검증 여부와 스텝 기반 검증 설정 추가
    validate_every_epoch = config['training'].get('validate_every_epoch', True)
    validate_every_n_steps = config['training'].get('validate_every_n_steps', 1000)
    save_every_n_steps = config['training'].get('save_every_n_steps', 1000)

    print("===== Training Start =====")
    global_step = 0
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
        for i, (wearing_img, product_img) in pbar:
            global_step += 1
            wearing_img = wearing_img.to(device, non_blocking=True)
            product_img = product_img.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(device_type='cuda', dtype=torch.float16):
                if method == 'simclr':
                    emb_wearing = model(wearing_img)
                    emb_product = model(product_img)
                    loss = criterion(emb_wearing, emb_product)
                    # 배치 단위 top-k accuracy 계산
                    with torch.no_grad():
                        batch_topk = compute_topk_accuracy(emb_wearing, emb_product)
                elif method == 'moco':
                    logits, labels = model(wearing_img, product_img)
                    loss = criterion(logits, labels)
                    # MoCo의 경우 queue를 포함한 정확한 계산을 위해 encode() 사용
                    with torch.no_grad():
                        q_emb = model.encode(wearing_img)
                        p_emb = model.encode(product_img)
                        batch_topk = compute_topk_accuracy(q_emb, p_emb)
            
            loss = loss / grad_acc_steps
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * grad_acc_steps
            
            pbar.set_postfix({
                "loss": f"{loss.item() * grad_acc_steps:.4f}",
                "top1": f"{batch_topk[1]:.1f}%"
            })
            if use_wandb:
                wandb.log({
                    "train/loss": loss.item() * grad_acc_steps,
                    "train/batch_top1": batch_topk[1],
                    "train/batch_top5": batch_topk[5],
                    "train/batch_top10": batch_topk[10],
                    "step": global_step,
                    "epoch": epoch+1
                })
            
            # -----------------------------
            # (NEW) 100번째 배치마다 시각화
            # -----------------------------
            if (i % 200 == 0) and (i > 0):
                current_vis_dir = os.path.join(vis_dir, f"epoch_{epoch+1}_step_{i}")
                os.makedirs(current_vis_dir, exist_ok=True)
                
                # Set visualization parameters
                distance_metric = 'euclidean'  # or 'euclidean'
                margin = None  # Set if you're using triplet or contrastive loss with margin
                
                if method == 'simclr':
                    # 1) Contrastive Distance Plot
                    save_contrastive_matrix(
                        emb_wearing, emb_product,
                        save_path=os.path.join(current_vis_dir, "dist_plot.png"),
                        distance_metric=distance_metric,
                        margin=margin
                    )
                    # 2) top-k image samples
                    save_topk_image_samples(
                        wearing_img, product_img,
                        emb_wearing, emb_product,
                        k=3,
                        distance_metric=distance_metric,
                        out_dir=current_vis_dir,
                    )
                elif method == 'moco':
                    # For MoCo, use the encoded embeddings
                    with torch.no_grad():
                        q_emb = model.encode(wearing_img)
                        p_emb = model.encode(product_img)
                        
                    # 1) Contrastive Distance Plot
                    save_contrastive_matrix(
                        q_emb, p_emb,
                        save_path=os.path.join(current_vis_dir, "dist_plot.png"),
                        distance_metric=distance_metric,
                        margin=margin
                    )
                    # 2) top-k image samples
                    save_topk_image_samples(
                        wearing_img, product_img,
                        q_emb, p_emb,
                        k=3,
                        distance_metric=distance_metric,
                        out_dir=current_vis_dir,
                    )
        
            # 스텝 기반 검증 추가
            if not validate_every_epoch and global_step % validate_every_n_steps == 0:
                print(f"\nValidating at step {global_step}...")
                model.eval()
                # 여기에 기존 검증 코드 추가 (원래 에폭 끝에서 실행되는 코드)
                all_query_emb = []
                all_candidate_emb = []
                with torch.no_grad():
                    for val_wearing, val_product in tqdm(val_loader, desc="[Validation]"):
                        val_wearing = val_wearing.to(device)
                        val_product = val_product.to(device)
                        if method == 'simclr':
                            q_emb = model(val_wearing)
                            p_emb = model(val_product)
                        elif method == 'moco':
                            q_emb = model.encode(val_wearing)
                            p_emb = model.encode(val_product)
                        all_query_emb.append(q_emb)
                        all_candidate_emb.append(p_emb)
                        
                query_emb = torch.cat(all_query_emb, dim=0)
                candidate_emb = torch.cat(all_candidate_emb, dim=0)
                topk_acc = compute_topk_accuracy(query_emb, candidate_emb, topk=(1, 5, 10))
                print(f"Validation Top1: {topk_acc[1]:.2f}% | Top5: {topk_acc[5]:.2f}% | Top10: {topk_acc[10]:.2f}%")
                if use_wandb:
                    wandb.log({"val/top1": topk_acc[1],
                               "val/top5": topk_acc[5],
                               "val/top10": topk_acc[10],
                               "step": global_step})
                
                # 체크포인트 저장
                if global_step % save_every_n_steps == 0:
                    ckpt_name = f"step_{global_step}.pth.tar"
                    ckpt_path = os.path.join(save_dir, ckpt_name)
                    save_checkpoint({
                        'epoch': epoch+1,
                        'step': global_step,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, ckpt_path)
                    print(f"Checkpoint saved: {ckpt_path}")
                
                model.train()  # 다시 학습 모드로 전환
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")
        
        # 에폭 기반 검증 (기존 코드)
        if validate_every_epoch:
            model.eval()
            all_query_emb = []
            all_candidate_emb = []
            with torch.no_grad():
                for val_wearing, val_product in tqdm(val_loader, desc="[Validation]"):
                    val_wearing = val_wearing.to(device)
                    val_product = val_product.to(device)
                    if method == 'simclr':
                        q_emb = model(val_wearing)
                        p_emb = model(val_product)
                    elif method == 'moco':
                        q_emb = model.encode(val_wearing)
                        p_emb = model.encode(val_product)
                    all_query_emb.append(q_emb)
                    all_candidate_emb.append(p_emb)
                    
            query_emb = torch.cat(all_query_emb, dim=0)
            candidate_emb = torch.cat(all_candidate_emb, dim=0)
            topk_acc = compute_topk_accuracy(query_emb, candidate_emb, topk=(1, 5, 10))
            print(f"Validation Top1: {topk_acc[1]:.2f}% | Top5: {topk_acc[5]:.2f}% | Top10: {topk_acc[10]:.2f}%")
            if use_wandb:
                wandb.log({"val/top1": topk_acc[1],
                           "val/top5": topk_acc[5],
                           "val/top10": topk_acc[10],
                           "epoch": epoch+1})
            
            # Epoch-level visualization (예: 전체 임베딩 분포, distance matrix 등)
            epoch_vis_path = os.path.join(vis_dir, f"epoch_{epoch+1}_overview.png")
            save_visualization(query_emb, candidate_emb, avg_loss, epoch_vis_path, mode="epoch")
            
            # Checkpoint 저장
            ckpt_name = f"epoch_{epoch+1}.pth.tar"
            ckpt_path = os.path.join(save_dir, ckpt_name)
            save_checkpoint({
                'epoch': epoch+1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")
            
            # Early Stopping based on Top1 accuracy
            if topk_acc[1] > best_val_top1:
                best_val_top1 = topk_acc[1]
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"No improvement for {epochs_no_improve} epoch(s).")
                if epochs_no_improve >= early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch+1}.")
                    break
        
    print("===== Training Complete =====")

if __name__ == '__main__':
    main()
