import os
import argparse
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler  # fallback
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import wandb
from timm.utils import ModelEmaV2

# 내부 모듈
from utils.config import load_config
from utils.dataset import ContrastiveFashionDataset, collate_fn_supcon
from utils.losses import HardNegSupConLoss  # 하드 네거티브 SupConLoss
from utils.metrics import compute_topk_accuracy
from utils.visualization import save_contrastive_matrix, save_topk_image_samples
from models.convnext import ConvNextModel

# 추가: ColorGroupSampler import
from utils.sampler import ColorGroupSampler

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
    alpha = config['loss'].get('hardneg_alpha', 2.0)  # 하드네거티브 가중치
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
    
    # 3. Dataset
    train_dataset = ContrastiveFashionDataset(
        root_dir=config['data']['train_dir'],
        metainfo_path=config['data']['train_metainfo_path'],
        color_group_path=config['data'].get('color_group_path', None),
        is_train=True,
        image_size=image_size,
        n_mask_channels=config['data'].get('n_mask_channels', 0),
        mode='supcon'
    )
    val_dataset = ContrastiveFashionDataset(
        root_dir=config['data']['val_dir'],
        metainfo_path=config['data']['val_metainfo_path'],
        color_group_path=config['data'].get('color_group_path', None),
        is_train=False,
        image_size=image_size,
        n_mask_channels=config['data'].get('n_mask_channels', 0),
        mode='supcon'
    )
    
    # 4. ColorGroupSampler로 train_loader 구성
    try:
        if len(train_dataset) < batch_size:
            print("Dataset size is smaller than batch_size. Using RandomSampler fallback.")
            train_sampler = RandomSampler(train_dataset)
        else:
            train_sampler = ColorGroupSampler(train_dataset, batch_size=batch_size, shuffle=True)
            print("Using ColorGroupSampler for training...")
    except Exception as e:
        # 상세한 에러 메시지 출력
        print(f"ColorGroupSampler failed with error: {str(e)}")
        print("Using RandomSampler fallback.")
        train_sampler = RandomSampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler if isinstance(train_sampler, ColorGroupSampler) else None,
        num_workers=num_workers,
        collate_fn=collate_fn_supcon
    )

    # val_loader는 일반적 방법 (shuffle=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=collate_fn_supcon
    )
    
    # 5. Model (ConvNeXt)
    model = ConvNextModel(backbone=backbone, pretrained=True, embed_dim=embed_dim).to(device)
    ema = ModelEmaV2(model, decay=0.999)
    
    # 6. Loss & Optimizer
    criterion = HardNegSupConLoss(temperature=temperature, alpha=alpha)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler = GradScaler()
    
    # 7. Resume Checkpoint (옵션)
    start_epoch = 0
    if resume and checkpoint_path:
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
    
    # 8. Training Loop
    print("===== Training Start =====")
    global_step = 0
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
        
        for i, (wearing_imgs, product_imgs, product_codes, color_groups) in pbar:
            global_step += wearing_imgs.shape[0]

            wearing_imgs = wearing_imgs.to(device, non_blocking=True)
            product_imgs = product_imgs.to(device, non_blocking=True)
            
            with autocast(device_type='cuda', dtype=torch.float16):
                emb_wearing = model(wearing_imgs)   # (B, D)
                emb_product = model(product_imgs)   # (B, D)
                embeddings = torch.cat([emb_wearing, emb_product], dim=0)  # (2B, D)

                loss = criterion(embeddings, product_codes, color_groups)
                loss = loss / grad_acc_steps
            
            scaler.scale(loss).backward()
            if (i + 1) % grad_acc_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                update_ema(ema, model)
            
            current_loss = loss.item() * grad_acc_steps
            running_loss += current_loss
            
            # In-batch retrieval 정확도
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
            
            # (선택) 100 배치마다 시각화
            if (i % 100 == 0) and (i > 0):
                current_vis_dir = os.path.join(vis_dir, f"epoch_{epoch+1}_step_{i}")
                os.makedirs(current_vis_dir, exist_ok=True)
                save_contrastive_matrix(
                    emb_wearing, emb_product, None,
                    save_path=os.path.join(current_vis_dir, "dist_plot.png"),
                    distance_metric='cosine', margin=None
                )
                save_topk_image_samples(
                    wearing_imgs, product_imgs,
                    emb_wearing, emb_product,
                    k=3, distance_metric='cosine',
                    out_dir=current_vis_dir
                )
        
        avg_loss = running_loss / len(train_loader)
        print(f"[Train] Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")
        if use_wandb:
            wandb.log({"train/Avg loss": avg_loss, "epoch": epoch+1})
        
        # ----------------- Validation -----------------
        model.eval()
        all_wearing_emb = []
        all_product_emb = []
        with torch.no_grad():
            for val_wearing, val_product, _, _ in tqdm(val_loader, desc="[Validation]"):
                val_wearing = val_wearing.to(device)
                val_product = val_product.to(device)
                
                emb_wearing = model(val_wearing)
                emb_product = model(val_product)
                
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
        
        # ----------------- Checkpoint -----------------
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
