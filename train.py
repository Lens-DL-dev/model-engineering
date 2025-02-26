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
from timm.utils import ModelEmaV2

# 내부 모듈
from utils.config import load_config
from utils.dataset import ContrastiveFashionDataset, collate_fn_supcon
from utils.losses import SupConLoss
from utils.metrics import compute_topk_accuracy
from utils.visualization import save_contrastive_matrix, save_topk_image_samples
from models.convnext import ConvNextModel

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

def update_ema(ema, model):
    if ema is not None:
        ema.update(model)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', type=str)
    args = parser.parse_args()
    
    # 1. Load Config
    config = load_config(args.config)
    use_wandb = config['use_wandb']
    if use_wandb:
        wandb.init(project="fashion-supcon", config=config)
        run_name = wandb.run.name
    else:
        run_name = datetime.now().strftime("%m%d%H%M%S")
        
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    lr = config['training']['learning_rate']
    num_workers = config['training']['num_workers']
    temperature = config['loss'].get('temperature', 0.07)
    save_dir = os.path.join(config['training'].get('save_dir', './checkpoints'), run_name)
    os.makedirs(save_dir, exist_ok=True)
    vis_dir = os.path.join(config['training'].get('vis_dir', './vis_results'), run_name)
    os.makedirs(vis_dir, exist_ok=True)
    grad_acc_steps = config['training'].get('gradient_accumulation_steps', 1)
    patience = config['training'].get('early_stopping_patience', 5)
    
    # 모델 설정
    backbone = config['model'].get('backbone', 'convnext_tiny')
    embed_dim = config['model'].get('embed_dim', 1024)
    checkpoint_path = config['model'].get('checkpoint', "")
    resume = config['model'].get('resume', False)
    image_size = config['model'].get('image_size', 224)
    in_channels = config['model'].get('in_channels', 3)
    
    # 2. Device
    if not torch.cuda.is_available():
        print("No CUDA device found. Exiting.")
        return
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
    
    # 3. Dataset / Dataloader (SupCon 모드)
    train_dataset = ContrastiveFashionDataset(
        root_dir=config['data']['train_dir'],
        metainfo_path=config['data']['train_metainfo_path'],
        is_train=True,
        image_size=image_size,
        n_mask_channels=config['data'].get('n_mask_channels', 0),
        mode='supcon'
    )
    val_dataset = ContrastiveFashionDataset(
        root_dir=config['data']['val_dir'],
        metainfo_path=config['data']['val_metainfo_path'],
        is_train=False,
        image_size=image_size,
        n_mask_channels=config['data'].get('n_mask_channels', 0),
        mode='supcon'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collate_fn_supcon
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=collate_fn_supcon
    )
    
    # 4. Model (ConvNeXt 기반)
    model = ConvNextModel(backbone=backbone, pretrained=True, embed_dim=embed_dim).to(device)
    ema = ModelEmaV2(model, decay=0.999)
    
    # 5. Loss & Optimizer
    criterion = SupConLoss(temperature=temperature)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler = GradScaler()
    
    # 6. Resume Checkpoint (옵션)
    start_epoch = 0
    if resume and checkpoint_path:
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
    
    # 7. Training Loop
    print("===== Training Start =====")
    global_step = 0
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
        
        for i, (wearing_imgs, product_imgs, product_codes) in pbar:
            global_step += wearing_imgs.shape[0]
            wearing_imgs = wearing_imgs.to(device, non_blocking=True)
            product_imgs = product_imgs.to(device, non_blocking=True)
            
            with autocast(device_type='cuda', dtype=torch.float16):
                emb_wearing = model(wearing_imgs)  # [B, D]
                emb_product = model(product_imgs)  # [B, D]
                # 두 view를 concat하여 (2B, D) 임베딩 생성
                embeddings = torch.cat([emb_wearing, emb_product], dim=0)
                loss = criterion(embeddings, product_codes)
                loss = loss / grad_acc_steps
            
            scaler.scale(loss).backward()
            if (i + 1) % grad_acc_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                update_ema(ema, model)
            
            current_loss = loss.item() * grad_acc_steps
            running_loss += current_loss
            
            # 배치 단위 retrieval 평가 (wearing_imgs vs product_imgs)
            with torch.no_grad():
                batch_topk = compute_topk_accuracy(emb_wearing, emb_product, topk=(1,5,10))
            pbar.set_postfix({
                "loss": f"{current_loss:.4f}",
                "train/top1": f"{batch_topk[1]:.2f}%",
                "train/top5": f"{batch_topk[5]:.2f}%",
                "train/top10": f"{batch_topk[10]:.2f}%"
            })
            if use_wandb:
                wandb.log({
                    "train/loss": current_loss,
                    "train/top1": batch_topk[1],
                    "train/top5": batch_topk[5],
                    "train/top10": batch_topk[10],
                    "step": global_step,
                    "epoch": epoch+1
                })
            
            # 100 배치마다 시각화
            if (i % 100 == 0) and (i > 0):
                current_vis_dir = os.path.join(vis_dir, f"epoch_{epoch+1}_step_{i}")
                os.makedirs(current_vis_dir, exist_ok=True)
                save_contrastive_matrix(emb_wearing, emb_product, None,
                                          save_path=os.path.join(current_vis_dir, "dist_plot.png"),
                                          distance_metric='cosine', margin=None)
                save_topk_image_samples(wearing_imgs, product_imgs, emb_wearing, emb_product,
                                        k=3, distance_metric='cosine', out_dir=current_vis_dir)
        
        avg_loss = running_loss / len(train_loader)
        print(f"[Train] Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")
        if use_wandb:
            wandb.log({"train/Avg loss": avg_loss, "epoch": epoch+1})
        
        # ------------- Validation -------------
        model.eval()
        all_wearing_emb = []
        all_product_emb = []
        with torch.no_grad():
            for val_wearing, val_product, _ in tqdm(val_loader, desc="[Validation]"):
                val_wearing = val_wearing.to(device)
                val_product = val_product.to(device)
                
                # 모델 출력 처리 방식 개선
                out_wearing = model(val_wearing)
                out_product = model(val_product)
                
                # 튜플 또는 리스트인 경우에만 인덱싱, 그렇지 않으면 그대로 사용
                emb_wearing = out_wearing[1] if isinstance(out_wearing, (list, tuple)) and len(out_wearing) > 1 else out_wearing
                emb_product = out_product[1] if isinstance(out_product, (list, tuple)) and len(out_product) > 1 else out_product
                
                all_wearing_emb.append(emb_wearing)
                all_product_emb.append(emb_product)
                
        query_emb = torch.cat(all_wearing_emb, dim=0)
        candidate_emb = torch.cat(all_product_emb, dim=0)
        topk_acc = compute_topk_accuracy(query_emb, candidate_emb, topk=(1,5,10))
        print(f"[Val] Epoch {epoch+1} - Top1: {topk_acc[1]:.2f}%, Top5: {topk_acc[5]:.2f}%, Top10: {topk_acc[10]:.2f}%")
        if use_wandb:
            wandb.log({
                "Val/top1": topk_acc[1],
                "Val/top5": topk_acc[5],
                "Val/top10": topk_acc[10],
                "epoch": epoch+1
            })
        
        # ------------- Checkpoint 저장 -------------
        ckpt_name = f"epoch_{epoch+1}.pth.tar"
        ckpt_path = os.path.join(save_dir, ckpt_name)
        save_checkpoint({
            'epoch': epoch+1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, ckpt_path)
        print(f"=> Checkpoint saved: {ckpt_path}")
        
        torch.cuda.empty_cache()
    
    print("===== Training Complete =====")

if __name__ == '__main__':
    main()
