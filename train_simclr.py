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
from utils.dataset import SimCLRFashionDataset, collate_fn_simclr
from utils.losses import SimCLRLoss
from utils.metrics import compute_topk_accuracy
from utils.visualization import save_contrastive_matrix, save_topk_image_samples
from models.convnext import ConvNextModel
from utils.optimizers import LARS

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
    parser.add_argument('--config', default='config_simclr.yaml', type=str)
    args = parser.parse_args()
    
    # 1. Load Config
    config = load_config(args.config)
    use_wandb = config['use_wandb']
    if use_wandb:
        wandb.init(project="fashion-simclr", config=config)
        run_name = wandb.run.name
    else:
        run_name = datetime.now().strftime("%m%d%H%M%S")
        
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    lr = config['training']['learning_rate']
    num_workers = config['training']['num_workers']
    temperature = config['loss'].get('temperature', 0.07)
    save_dir = os.path.join(config['training'].get('save_dir', './checkpoints_simclr'), run_name)
    os.makedirs(save_dir, exist_ok=True)
    vis_dir = os.path.join(config['training'].get('vis_dir', './vis_results_simclr'), run_name)
    os.makedirs(vis_dir, exist_ok=True)
    grad_acc_steps = config['training'].get('gradient_accumulation_steps', 1)
    patience = config['training'].get('early_stopping_patience', 5)
    
    # 모델 설정
    backbone = config['model'].get('backbone', 'convnext_tiny')
    embed_dim = config['model'].get('embed_dim', 1024)
    checkpoint_path = config['model'].get('checkpoint', "")
    resume = config['model'].get('resume', False)
    image_size = config['model'].get('image_size', 224)
    
    # 2. Device
    if not torch.cuda.is_available():
        print("No CUDA device found. Exiting.")
        return
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
    
    # 3. Dataset / Dataloader (SimCLR 모드)
    train_dataset = SimCLRFashionDataset(
        root_dir=config['data']['train_dir'],
        metainfo_path=config['data'].get('train_metainfo_path', None),
        is_train=True,
        image_size=image_size,
        max_samples=config['data'].get('max_samples', -1)
    )
    
    val_dataset = SimCLRFashionDataset(
        root_dir=config['data']['val_dir'],
        metainfo_path=config['data'].get('val_metainfo_path', None),
        is_train=False,
        image_size=image_size,
        max_samples=config['data'].get('val_max_samples', -1)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collate_fn_simclr
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )
    
    # 4. Model (ConvNeXt 기반)
    model = ConvNextModel(backbone=backbone, pretrained=True, embed_dim=embed_dim).to(device)
    ema = ModelEmaV2(model, decay=0.999)
    
    # 5. Loss & Optimizer
    criterion = SimCLRLoss(temperature=temperature)
    optimizer = LARS(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=1e-5,
        trust_coefficient=0.001,
        exclude_bias_and_norm=True
    )
    scaler = GradScaler()
    
    # 6. Resume Checkpoint (옵션)
    start_epoch = 0
    if resume and checkpoint_path:
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
    
    # 7. Training Loop
    print("===== SimCLR Training Start =====")
    global_step = 0
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
        
        for i, (aug_imgs1, aug_imgs2, _) in pbar:
            global_step += aug_imgs1.shape[0]
            aug_imgs1 = aug_imgs1.to(device, non_blocking=True)
            aug_imgs2 = aug_imgs2.to(device, non_blocking=True)
            
            with autocast(device_type='cuda', dtype=torch.float16):
                # 두 augmentation view의 임베딩 추출
                emb_view1 = model(aug_imgs1)  # [B, D]
                emb_view2 = model(aug_imgs2)  # [B, D]
                
                # 두 view를 concat하여 (2B, D) 임베딩 생성
                embeddings = torch.cat([emb_view1, emb_view2], dim=0)
                
                # SimCLR 손실 계산 (레이블 정보 없이)
                loss = criterion(embeddings)
                loss = loss / grad_acc_steps
            
            scaler.scale(loss).backward()
            if (i + 1) % grad_acc_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                update_ema(ema, model)
            
            current_loss = loss.item() * grad_acc_steps
            running_loss += current_loss
            
            pbar.set_postfix({
                "loss": f"{current_loss:.4f}"
            })
            
            if use_wandb:
                wandb.log({
                    "train/loss": current_loss,
                    "step": global_step,
                    "epoch": epoch+1
                })
            
            # 100 배치마다 시각화
            if (i % 100 == 0) and (i > 0):
                current_vis_dir = os.path.join(vis_dir, f"epoch_{epoch+1}_step_{i}")
                os.makedirs(current_vis_dir, exist_ok=True)
                save_contrastive_matrix(emb_view1, emb_view2, None,
                                        save_path=os.path.join(current_vis_dir, "dist_plot.png"),
                                        distance_metric='cosine', margin=None)
        
        avg_loss = running_loss / len(train_loader)
        print(f"[Train] Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")
        if use_wandb:
            wandb.log({"train/avg_loss": avg_loss, "epoch": epoch+1})
        
        # Checkpoint 저장
        ckpt_name = f"epoch_{epoch+1}.pth.tar"
        ckpt_path = os.path.join(save_dir, ckpt_name)
        save_checkpoint({
            'epoch': epoch+1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, ckpt_path)
        print(f"=> Checkpoint saved: {ckpt_path}")
        
        torch.cuda.empty_cache()
    
    print("===== SimCLR Training Complete =====")

if __name__ == '__main__':
    main() 