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
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.optim.lr_scheduler import SequentialLR
import numpy as np
import gc
from collections import defaultdict
from torch.optim import AdamW

# 내부 모듈
from utils.config import load_config
from utils.dataset import ContrastiveFashionDataset, collate_fn_supcon
from utils.losses import SupConLoss, MarginSupConLoss
from utils.metrics import compute_topk_accuracy
from utils.visualization import save_contrastive_matrix, save_topk_image_samples
from models.convnext import ConvNextModel
from utils.optimizers import LARS
from utils.memory_bank import MemoryBank

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def load_checkpoint(model, optimizer, filename):
    if os.path.isfile(filename):
        print(f"=> Loading checkpoint from '{filename}'")
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

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def log_gradients_in_model(model, step, prefix="gradients"):
    """WandB에 그라디언트 히스토그램 로깅"""
    if wandb.run is None:
        return
    
    # 그라디언트 히스토그램 로깅
    for tag, value in model.named_parameters():
        if value.grad is not None:
            wandb.log({f"{prefix}/{tag}": wandb.Histogram(value.grad.cpu().numpy())}, step=step)

def log_gpu_memory():
    """현재 GPU 메모리 사용량 로깅"""
    if wandb.run is None or not torch.cuda.is_available():
        return
    
    # GPU 메모리 사용량 로깅
    memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
    memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # GB
    wandb.log({
        "gpu_memory/allocated_gb": memory_allocated,
        "gpu_memory/reserved_gb": memory_reserved
    })

def log_embedding_samples(embeddings, labels, step, title="embeddings"):
    """WandB에 임베딩 샘플 시각화 로깅"""
    if wandb.run is None or embeddings.shape[0] == 0:
        return
    
    # 최대 200개 샘플만 사용
    max_samples = min(200, embeddings.shape[0])
    embedding_data = embeddings[:max_samples].cpu().numpy()
    labels_data = labels[:max_samples]
    
    # Table을 통한 임베딩 데이터 로깅
    columns = ["label"] + [f"d{i}" for i in range(embedding_data.shape[1])]
    data = [[labels_data[i]] + embedding_data[i].tolist() for i in range(len(labels_data))]
    
    wandb.log({
        f"{title}_table": wandb.Table(
            columns=columns,
            data=data
        )
    }, step=step)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', type=str)
    args = parser.parse_args()
    
    # 1. Load Config
    config = load_config(args.config)
    
    # WandB 설정
    wandb_config = config.get('wandb', {})
    use_wandb = wandb_config.get('enabled', config.get('use_wandb', False))
    if use_wandb:
        wandb.init(
            project=wandb_config.get('project_name', "fashion-supcon-a100"),
            tags=wandb_config.get('tags', ["a100", "supcon", "memory-bank"]),
            config=config,
            name=wandb_config.get('run_name'),
        )
        run_name = wandb.run.name
    else:
        run_name = datetime.now().strftime("%m%d%H%M%S")
    
    # 훈련 설정 추출
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    lr = config['training']['learning_rate']
    num_workers = config['training']['num_workers']
    temperature = config['loss'].get('temperature', 0.07)
    hard_mining = config['loss'].get('hard_mining', True)
    save_dir = os.path.join(config['training'].get('save_dir', './checkpoints'), run_name)
    os.makedirs(save_dir, exist_ok=True)
    vis_dir = os.path.join(config['training'].get('vis_dir', './vis_results'), run_name)
    os.makedirs(vis_dir, exist_ok=True)
    grad_acc_steps = config['training'].get('gradient_accumulation_steps', 1)
    patience = config['training'].get('early_stopping_patience', 5)
    eval_interval = config['training'].get('eval_interval', 1)
    save_interval = config['training'].get('save_interval', 1)
    mixed_precision = config['training'].get('mixed_precision', True)
    visualization_interval = config['training'].get('visualization_interval', 100)
    log_interval = config['training'].get('log_interval', 10)
    
    # 모델 설정 - 단일 ConvNextModel 사용으로 수정
    backbone = config['model'].get('backbone', 'convnext_tiny')
    embed_dim = config['model'].get('embed_dim', 512)
    checkpoint_path = config['model'].get('checkpoint', "")
    resume = config['model'].get('resume', False)
    image_size = config['model'].get('image_size', 384)
    in_channels = config['model'].get('in_channels', 3)
    
    # 메모리 뱅크 설정
    memory_bank_config = config.get('memory_bank', {})
    use_memory_bank = memory_bank_config.get('enabled', True)
    memory_bank_size = memory_bank_config.get('size', 16384)
    memory_bank_momentum = memory_bank_config.get('momentum', 0.99)
    memory_bank_start_epoch = memory_bank_config.get('start_epoch', 1)
    
    # 2. Device
    if not torch.cuda.is_available():
        print("No CUDA device found. Exiting.")
        return
    
    # 사용 가능한 모든 GPU 확인
    n_gpus = torch.cuda.device_count()
    print(f"Found {n_gpus} GPU(s)")
    
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
    
    # GPU 정보 출력
    if n_gpus > 0:
        for i in range(n_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"GPU {i}: {gpu_name}, Memory: {gpu_mem:.1f} GB")
    
    # 3. Dataset / Dataloader (SupCon 모드)
    augmentation_config = config.get('augmentation', {})
    
    train_dataset = ContrastiveFashionDataset(
        root_dir=config['data']['train_dir'],
        metainfo_path=config['data']['train_metainfo_path'],
        is_train=True,
        image_size=image_size,
        n_mask_channels=config['data'].get('n_mask_channels', 0),
        mode='supcon',
        config=config  # augmentation 설정 전달
    )
    val_dataset = ContrastiveFashionDataset(
        root_dir=config['data']['val_dir'],
        metainfo_path=config['data']['val_metainfo_path'],
        is_train=False,
        image_size=image_size,
        n_mask_channels=config['data'].get('n_mask_channels', 0),
        mode='supcon',
        config=config
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collate_fn_supcon,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=collate_fn_supcon,
        pin_memory=True
    )
    
    # 4. 단일 ConvNextModel 모델 초기화
    model = ConvNextModel(
        backbone=backbone,
        pretrained=True,
        embed_dim=embed_dim
    ).to(device)
    
    ema = ModelEmaV2(model, decay=0.999)
    
    # WandB에 모델 아키텍처 로깅
    if use_wandb and wandb_config.get('log_model', True):
        wandb.watch(model, log="all", log_freq=100)
    
    # 5. Loss & Optimizer
    criterion = MarginSupConLoss(temperature=temperature, margin=0.3, hard_mining=True)
    
    # LARS 대신 AdamW 사용
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=float(config['optimizer'].get('weight_decay', 1e-5))
    )
    
    # Mixed precision 설정
    scaler = GradScaler() if mixed_precision else None
    
    # 6. Resume Checkpoint (옵션)
    start_epoch = 0
    if resume and checkpoint_path:
        checkpoint_epoch = load_checkpoint(model, optimizer, checkpoint_path)
        print(f"모델 가중치는 에폭 {checkpoint_epoch}에서 로드했지만, 에폭 카운터는 0으로 재설정합니다.")
        # start_epoch = checkpoint_epoch  # 이 줄을 주석 처리하거나 삭제
    
    # 7. 메모리 뱅크 초기화
    memory_bank = MemoryBank(
        size=memory_bank_size, 
        dim=embed_dim, 
        device=device,
        momentum=memory_bank_momentum
    )
    
    # 8. 웜업 + 코사인 어닐링 스케줄러 설정
    warmup_epochs = config['training'].get('warmup_epochs', 5)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, 
                               total_iters=warmup_epochs * len(train_loader))
    cosine_scheduler = CosineAnnealingLR(optimizer, 
                                        T_max=(epochs - warmup_epochs) * len(train_loader),
                                        eta_min=1e-6)
    scheduler = SequentialLR(optimizer, 
                             schedulers=[warmup_scheduler, cosine_scheduler],
                             milestones=[warmup_epochs * len(train_loader)])
    
    # 9. 학습 과정 트래킹 & 조기 종료 설정
    best_top1 = 0.0
    best_epoch = 0
    patience_counter = 0
    
    # 10. 학습 시작
    print("\n===== Training Start =====")
    print(f"Model: {backbone}, Embed dim: {embed_dim}, Batch size: {batch_size}")
    print(f"Learning rate: {lr}, Temperature: {temperature}")
    print(f"Mixed precision: {mixed_precision}, Memory bank: {use_memory_bank} (size: {memory_bank_size})")
    print(f"Epochs: {epochs}, Warmup epochs: {warmup_epochs}, Grad acc steps: {grad_acc_steps}")
    print(f"Image size: {image_size}")
    print("==============================\n")
    
    global_step = 0
    print(f"Starting training from epoch {start_epoch} to {epochs}")
    print(f"Memory bank enabled: {use_memory_bank}, size: {memory_bank_size}")
    print(f"Using criterion: {criterion.__class__.__name__}")

    # 첫 배치 처리 시작 전 로그 추가
    print("Processing first batch...")
    
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        running_metrics = defaultdict(float)
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")
        
        # 에폭 시작 시 메모리 로깅
        if use_wandb and wandb_config.get('log_memory', True):
            log_gpu_memory()
        
        # 메모리 뱅크 사용 여부 결정 (지정된 에폭부터 사용)
        use_memory_bank_this_epoch = use_memory_bank and epoch >= memory_bank_start_epoch
        
        for i, (wearing_imgs, product_imgs, product_codes) in pbar:
            # 배치 크기 측정
            batch_size_actual = wearing_imgs.shape[0]
            global_step += batch_size_actual
            
            wearing_imgs = wearing_imgs.to(device, non_blocking=True)
            product_imgs = product_imgs.to(device, non_blocking=True)
            
            with autocast(device_type='cuda', enabled=mixed_precision):
                # 단일 모델로 각 이미지를 별도로 처리 (원래 old_train 방식과 동일)
                emb_wearing = model(wearing_imgs)  # [B, D]
                emb_product = model(product_imgs)  # [B, D]
                
                # 두 view를 concat하여 (2B, D) 임베딩 생성
                embeddings = torch.cat([emb_wearing, emb_product], dim=0)
                
                # 메모리 뱅크 활용
                if use_memory_bank_this_epoch:
                    memory_embs, memory_codes = memory_bank.get()
                    if len(memory_codes) > 0:
                        loss = criterion(embeddings, product_codes, memory_embs, memory_codes)
                    else:
                        loss = criterion(embeddings, product_codes)
                else:
                    loss = criterion(embeddings, product_codes)
                
                loss = loss / grad_acc_steps
            
            # 역전파
            if mixed_precision:
                scaler.scale(loss).backward()
                if (i + 1) % grad_acc_steps == 0 or (i + 1) == len(train_loader):
                    # 그라디언트 로깅 (선택적)
                    if use_wandb and wandb_config.get('log_gradients', False) and (global_step % 500) == 0:
                        log_gradients_in_model(model, global_step)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()  # 스케줄러 업데이트
                    optimizer.zero_grad(set_to_none=True)
                    update_ema(ema, model)
            else:
                loss.backward()
                if (i + 1) % grad_acc_steps == 0 or (i + 1) == len(train_loader):
                    # 그라디언트 로깅 (선택적)
                    if use_wandb and wandb_config.get('log_gradients', False) and (global_step % 500) == 0:
                        log_gradients_in_model(model, global_step)
                    
                    optimizer.step()
                    scheduler.step()  # 스케줄러 업데이트
                    optimizer.zero_grad(set_to_none=True)
                    update_ema(ema, model)
            
            # 메모리 뱅크 업데이트 (product_imgs의 임베딩만 사용)
            if use_memory_bank:
                with torch.no_grad():
                    memory_bank.enqueue_and_dequeue(F.normalize(emb_product, dim=1), product_codes)
            
            current_loss = loss.item() * grad_acc_steps
            running_loss += current_loss
            
            # 배치 단위 retrieval 평가 (wearing_imgs vs product_imgs)
            with torch.no_grad():
                batch_topk = compute_topk_accuracy(emb_wearing, emb_product, topk=(1,5,10))
                for k, v in batch_topk.items():
                    running_metrics[f"top{k}"] += v * batch_size_actual
            
            current_lr = get_lr(optimizer)
            
            # 프로그레스 바 업데이트
            pbar.set_postfix({
                "loss": f"{current_loss:.4f}",
                "top1": f"{batch_topk[1]:.2f}%",
                "top5": f"{batch_topk[5]:.2f}%",
                "lr": f"{current_lr:.6f}"
            })
            
            # WandB 로깅 (배치 단위)
            if use_wandb and (i % log_interval == 0 or i == len(train_loader) - 1):
                log_dict = {
                    "train/loss": current_loss,
                    "train/top1": batch_topk[1],
                    "train/top5": batch_topk[5],
                    "train/top10": batch_topk[10],
                    "train/lr": current_lr,
                    "train/memory_bank_size": memory_bank.get_size() if use_memory_bank else 0,
                    "train/epoch": epoch + (i / len(train_loader)),
                    "train/step": global_step,
                }
                wandb.log(log_dict, step=global_step)
            
            # 주기적 시각화
            if (i % visualization_interval == 0) and (i > 0):
                current_vis_dir = os.path.join(vis_dir, f"epoch_{epoch+1}_step_{i}")
                os.makedirs(current_vis_dir, exist_ok=True)
                
                with torch.no_grad():
                    save_contrastive_matrix(
                        emb_wearing, emb_product, None,
                        save_path=os.path.join(current_vis_dir, "dist_plot.png"),
                        distance_metric='cosine', margin=None
                    )
                    save_topk_image_samples(
                        wearing_imgs, product_imgs, emb_wearing, emb_product,
                        k=5, distance_metric='cosine', out_dir=current_vis_dir
                    )
        
        # 에폭 단위 메트릭 계산
        samples_seen = len(train_loader) * batch_size
        avg_loss = running_loss / len(train_loader)
        avg_metrics = {k: v / samples_seen for k, v in running_metrics.items()}
        
        print(f"[Train] Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}, ", end="")
        print(f"Top1: {avg_metrics['top1']:.2f}%, Top5: {avg_metrics['top5']:.2f}%")
        
        # WandB 에폭 단위 로깅
        if use_wandb:
            wandb.log({
                "epoch/train_loss": avg_loss,
                "epoch/train_top1": avg_metrics['top1'],
                "epoch/train_top5": avg_metrics['top5'],
                "epoch/epoch": epoch+1,
            }, step=global_step)
        
        # ------------- Validation -------------
        if (epoch + 1) % eval_interval == 0:
            model.eval()
            val_loss = 0.0
            all_wearing_emb = []
            all_product_emb = []
            all_product_codes = []
            
            with torch.no_grad():
                for val_wearing, val_product, val_codes in tqdm(val_loader, desc="[Validation]"):
                    val_wearing = val_wearing.to(device)
                    val_product = val_product.to(device)
                    
                    # 단일 모델로 각 이미지를 별도로 처리
                    emb_wearing = model(val_wearing)
                    emb_product = model(val_product)
                    
                    # SupCon Loss 계산
                    embeddings = torch.cat([emb_wearing, emb_product], dim=0)
                    loss = criterion(embeddings, val_codes)
                    val_loss += loss.item()
                    
                    all_wearing_emb.append(emb_wearing)
                    all_product_emb.append(emb_product)
                    all_product_codes.extend(val_codes)
                
            # 전체 데이터셋에 대한 메트릭 계산
            query_emb = torch.cat(all_wearing_emb, dim=0)
            candidate_emb = torch.cat(all_product_emb, dim=0)
            val_loss /= len(val_loader)
            
            topk_acc = compute_topk_accuracy(query_emb, candidate_emb, topk=(1,5,10))
            print(f"[Val] Epoch {epoch+1} - Loss: {val_loss:.4f}, ", end="")
            print(f"Top1: {topk_acc[1]:.2f}%, Top5: {topk_acc[5]:.2f}%, Top10: {topk_acc[10]:.2f}%")
            
            # WandB 검증 로깅
            if use_wandb:
                wandb.log({
                    "epoch/val_loss": val_loss,
                    "epoch/val_top1": topk_acc[1],
                    "epoch/val_top5": topk_acc[5],
                    "epoch/val_top10": topk_acc[10],
                    "epoch/epoch": epoch+1,
                }, step=global_step)
            
            # 조기 종료 및 최고 성능 모델 저장
            if topk_acc[1] > best_top1:
                best_top1 = topk_acc[1]
                best_epoch = epoch + 1
                patience_counter = 0
                
                # 최고 성능 모델 저장
                best_ckpt_path = os.path.join(save_dir, "best_model.pth.tar")
                save_checkpoint({
                    'epoch': epoch+1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_top1': best_top1,
                }, best_ckpt_path)
                print(f"=> New best model saved (Top1: {best_top1:.2f}%)")
                
                # EMA 모델도 저장
                if ema is not None:
                    ema_ckpt_path = os.path.join(save_dir, "best_model_ema.pth.tar")
                    save_checkpoint({
                        'epoch': epoch+1,
                        'state_dict': ema.module.state_dict(),
                        'best_top1': best_top1,
                    }, ema_ckpt_path)
            else:
                patience_counter += 1
                print(f"=> No improvement for {patience_counter} epochs (Best Top1: {best_top1:.2f}% at epoch {best_epoch})")
                
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {patience} epochs without improvement")
                    break
        
        # ------------- Checkpoint 저장 -------------
        if (epoch + 1) % save_interval == 0:
            ckpt_name = f"epoch_{epoch+1}.pth.tar"
            ckpt_path = os.path.join(save_dir, ckpt_name)
            save_checkpoint({
                'epoch': epoch+1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, ckpt_path)
            print(f"=> Checkpoint saved: {ckpt_path}")
            
            # WandB에 모델 아티팩트 저장 (선택적)
            if use_wandb and wandb_config.get('log_model', True) and (epoch + 1) % 5 == 0:
                wandb.save(ckpt_path, base_path=os.path.dirname(ckpt_path))
        
        # 메모리 정리
        torch.cuda.empty_cache()
        gc.collect()
    
    print("\n===== Training Complete =====")
    print(f"Best Top1: {best_top1:.2f}% at epoch {best_epoch}")
    
    # 최종 결과 WandB에 로깅
    if use_wandb:
        wandb.run.summary["best_top1"] = best_top1
        wandb.run.summary["best_epoch"] = best_epoch
        wandb.finish()

if __name__ == '__main__':
    main()
