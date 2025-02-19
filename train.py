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
from utils.visualization import save_visualization

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
    
    train_dataset = ContrastiveFashionDataset(root_dir=train_dir,
                                               metainfo_path=train_metainfo,
                                               is_train=True,
                                               image_size=image_size,
                                               n_mask_channels=n_mask_channels)
    val_dataset = ContrastiveFashionDataset(root_dir=val_dir,
                                             metainfo_path=val_metainfo,
                                             is_train=False,
                                             image_size=image_size,
                                             n_mask_channels=n_mask_channels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, drop_last=False)
    
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
                elif method == 'moco':
                    # MoCo: query=wearing, key=product
                    logits, labels = model(wearing_img, product_img)
                    loss = criterion(logits, labels)
            loss = loss / grad_acc_steps
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * grad_acc_steps
            
            pbar.set_postfix({"loss": f"{loss.item() * grad_acc_steps:.4f}"})
            if use_wandb:
                wandb.log({"train/loss": loss.item() * grad_acc_steps,
                           "step": global_step,
                           "epoch": epoch+1})
            
            # 100번째 배치마다 visualization (예: 샘플 이미지와 임베딩 분포)
            if (i + 1) % 100 == 0:
                vis_path = os.path.join(vis_dir, f"epoch_{epoch+1}_batch_{i+1}.png")
                # save_visualization 함수는 (wearing_img, product_img, loss, 임베딩 등) 시각화 이미지를 저장합니다.
                # 아래는 예시 호출이며, 실제 구현에 맞게 수정할 수 있습니다.
                save_visualization(wearing_img, product_img, loss.item() * grad_acc_steps, vis_path)
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")
        
        # Epoch마다 Validation 수행
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
                    # For evaluation, use encode() function from MoCoModel (should be defined in the model)
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
