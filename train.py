import os
import argparse
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast  # 수정됨
from tqdm import tqdm
import wandb
from transformers import CLIPVisionModel
from timm.utils import ModelEmaV2


# 내부 모듈
from utils.config import load_config
from utils.dataset import ContrastiveFashionDataset, collate_fn_contrastive
from utils.losses import ContrastiveLoss
from utils.metrics import calculate_tp_fp_tn_fn, compute_f1_score
from utils.visualization import save_contrastive_matrix, save_topk_image_samples, save_topk_image_samples_with_mask
from models.efficientnet_v2 import EfficientNetV2L

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

    # -----------------------------
    # 1. Load Config
    # -----------------------------
    config = load_config(args.config)
    use_wandb = config['use_wandb']
    if use_wandb:
        wandb.init(
            project="dev-lens",
            config=config
        )
        run_name = wandb.run.name
    else:
        run_name = datetime.now().strftime("%m%d%H%M%S")
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    lr = config['training']['learning_rate']
    num_workers = config['training']['num_workers']
    margin = config['loss'].get('margin', 1.0)
    distance_metric = config['loss'].get('distance_metric', 'euclidean')
    save_dir = config['training'].get('save_dir', './checkpoints')
    save_dir = os.path.join(save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    vis_dir = config['training'].get('vis_dir', './vis_results')
    vis_dir = os.path.join(vis_dir, run_name)
    os.makedirs(vis_dir, exist_ok=True)

    grad_acc_steps = config['training'].get('gradient_accumulation_steps', 1)

    # early stopping
    patience = config['training'].get('early_stopping_patience', 5)
    best_val_acc = 0.0
    epochs_no_improve = 0

    # checkpoint
    pretrained_model_name = config['model']['pretrained_model_name']
    checkpoint_path = config['model']['checkpoint']  
    resume = config['model'].get('resume', False)
    image_size = config['model'].get('image_size', 224)
    #embed_dim = config['model'].get('embed_dim', 512)
    in_channels = config['model'].get('in_channels', 3)


    # -----------------------------
    # 2. Device
    # -----------------------------
    if not torch.cuda.is_available():
        print("No CUDA device found. Exiting.")
        return
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True

    # -----------------------------
    # 3. Dataset / Dataloader
    # -----------------------------
    train_dir = config['data']['train_dir']
    val_dir = config['data']['val_dir']

    negative_count = config['data'].get('negative_count', 4)
    n_mask_channels = config['data'].get('n_mask_channels', 0)
    train_metainfo_path =  config['data'].get('train_metainfo_path', os.path.join(train_dir, 'metainfo.json'))
    val_metainfo_path =  config['data'].get('val_metainfo_path', os.path.join(val_dir, 'metainfo.json'))
    assert n_mask_channels + 3 == in_channels, f'in_channels == {in_channels}, But n_mask_channels == {n_mask_channels}. 모델의 입력 채널은 3(RGB) + N(mask)여야함!'
    train_dataset = ContrastiveFashionDataset(
        root_dir=train_dir,
        metainfo_path=train_metainfo_path,
        is_train=True,
        image_size=image_size,
        negative_count=negative_count,
        n_mask_channels=n_mask_channels
    )
    val_dataset = ContrastiveFashionDataset(
        root_dir=val_dir,
        metainfo_path=val_metainfo_path,
        is_train=False,
        image_size=image_size,
        negative_count=negative_count,
        n_mask_channels=n_mask_channels
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,   
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collate_fn_contrastive
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=collate_fn_contrastive
    )

    # -----------------------------
    # 4. Model
    # -----------------------------
    #model = EfficientNetV2L(pretrained=True, embed_dim=embed_dim, in_channels=in_channels).to(device)
    model = CLIPVisionModel.from_pretrained(pretrained_model_name).to(device)
    ema = ModelEmaV2(model, decay=0.999)# add EMA
    # -----------------------------
    # 5. Loss & Optimizer
    # -----------------------------
    criterion = ContrastiveLoss(margin=margin, distance_metric=distance_metric)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    # AMP -> Fix the future warning
    #   "torch.cuda.amp.GradScaler(args...) is deprecated. Please use torch.amp.GradScaler('cuda', args...) instead."
    scaler = torch.cuda.amp.GradScaler()

    # -----------------------------
    # 6. (Optional) Resume Checkpoint
    # -----------------------------
    start_epoch = 0
    if resume and checkpoint_path:
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path)

    # -----------------------------
    # 7. Training Loop
    # -----------------------------
    print("===== Training Start =====")
    global_step = 0

    for epoch in range(0, epochs):
        model.train()
        running_loss = 0.0
        
        # (개선) 에폭 전체 정확도 계산 위해
        epoch_tp = 0
        epoch_tn = 0
        epoch_fp = 0
        epoch_fn = 0

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
        for i, (anchors, candidates, labels) in pbar:
            global_step += anchors.shape[0]

            anchors = anchors.to(device, non_blocking=True)
            candidates = candidates.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # AMP
            with autocast(device_type='cuda', dtype=torch.float16):
                emb_anchor = model(anchors)[1]
                emb_candidate = model(candidates)[1]
                loss = criterion(emb_anchor, emb_candidate, labels)
                loss = loss / grad_acc_steps

            scaler.scale(loss).backward()

            if (i + 1) % grad_acc_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                update_ema(ema, model)    

            current_loss = loss.item() * grad_acc_steps
            running_loss += current_loss

            # (개선) 배치 내 correct/total
            with torch.no_grad():
                # 배치별 Accuracy
                tp, fp, tn, fn = calculate_tp_fp_tn_fn(
                    emb_anchor, emb_candidate, labels,
                    threshold=margin,
                    distance_metric=distance_metric
                )
                epoch_tp += tp
                epoch_fp += fp
                epoch_tn += tn
                epoch_fn += fn

            pbar.set_postfix({
                "loss": f"{current_loss:.4f}",
            })
            if use_wandb:
                wandb.log({"train/loss": current_loss, "step": global_step, "epoch": epoch + 1})
            
            # -----------------------------
            # (NEW) 300번째 배치마다 시각화
            # -----------------------------
            if (i % 300 == 0) and (i > 0):
                current_vis_dir = os.path.join(vis_dir, f"epoch_{epoch+1}_step_{i}")
                os.makedirs(current_vis_dir, exist_ok=True)

                # 1) Contrastive Distance Plot
                save_contrastive_matrix(
                    emb_anchor, emb_candidate, labels,
                    save_path=os.path.join(current_vis_dir, "dist_plot.png"),
                    distance_metric=distance_metric,
                    margin=margin
                )
                # 2) top-k 이미지 samples
                if n_mask_channels == 0:
                    save_topk_image_samples(
                    anchors, candidates,
                    emb_anchor, emb_candidate,
                    k=3,
                    distance_metric=distance_metric,
                    out_dir=current_vis_dir,
                )
                elif n_mask_channels == 1:
                    save_topk_image_samples_with_mask(
                        anchors, candidates,
                        emb_anchor, emb_candidate,
                        k=3,
                        distance_metric=distance_metric,
                        out_dir=current_vis_dir,
                    )
                else:
                    raise NotImplementedError

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = ((epoch_tp+epoch_tn) / (epoch_tp+epoch_tn+epoch_fp+epoch_fn)) * 100.0
        epoch_f1 = compute_f1_score(epoch_tp, epoch_fp, epoch_tn, epoch_fn)
        print(f"[Train] Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%, F1: {epoch_f1:.4f}")
        if use_wandb:
            wandb.log({"train/Avg loss": current_loss, "train/Avg acc": epoch_acc, "train/Avg f1": epoch_f1, "epoch": epoch + 1})
        
        del anchors, candidates, emb_anchor, emb_candidate, loss
        # -----------------------------
        # Validation
        # -----------------------------
        model.eval()
        val_tp = 0
        val_fp = 0
        val_tn = 0
        val_fn = 0
        val_loss_sum = 0.0

        with torch.no_grad():
            for val_i, (val_anchors, val_candidates, val_labels) in enumerate(tqdm(val_loader, desc="[Val]")):
                val_anchors = val_anchors.to(device)
                val_candidates = val_candidates.to(device)
                val_labels = val_labels.to(device)

                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    val_emb_anchor = model(val_anchors)[1]
                    val_emb_candidate = model(val_candidates)[1]
                    val_loss = criterion(val_emb_anchor, val_emb_candidate, val_labels)
                    val_loss_sum += val_loss.item()

                tp, fp, tn, fn = calculate_tp_fp_tn_fn(
                    val_emb_anchor, val_emb_candidate, val_labels,
                    threshold=margin,
                    distance_metric=distance_metric
                )
                val_tp += tp
                val_fp += fp
                val_tn += tn
                val_fn += fn

        val_loss_avg = val_loss_sum / max(len(val_loader), 1)
        val_acc = ((val_tp+val_tn) / (val_tp + val_fp + val_tn + val_fn)) * 100.0
        val_f1 = compute_f1_score(val_tp, val_fp, val_tn, val_fn)
        print(f"[Val] Epoch {epoch+1} - Loss: {val_loss_avg:.4f}, Acc: {val_acc:.2f}%, F1: {val_f1:.4f}")
        if use_wandb:
            wandb.log({"Val/Avg loss": val_loss_avg, "Val/Avg acc": val_acc, "Val/Avg f1": val_f1, "epoch": epoch + 1})
        
        
        # -----------------------------
        # EMA Validation
        # -----------------------------
        ema_model = ema.module
        ema_model.eval()
        val_tp = 0
        val_fp = 0
        val_tn = 0
        val_fn = 0
        val_loss_sum = 0.0

        with torch.no_grad():
            for val_i, (val_anchors, val_candidates, val_labels) in enumerate(tqdm(val_loader, desc="[EMA]")):
                val_anchors = val_anchors.to(device)
                val_candidates = val_candidates.to(device)
                val_labels = val_labels.to(device)

                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    val_emb_anchor = ema_model(val_anchors)[1]
                    val_emb_candidate = ema_model(val_candidates)[1]
                    val_loss = criterion(val_emb_anchor, val_emb_candidate, val_labels)
                    val_loss_sum += val_loss.item()

                tp, fp, tn, fn = calculate_tp_fp_tn_fn(
                    val_emb_anchor, val_emb_candidate, val_labels,
                    threshold=margin,
                    distance_metric=distance_metric
                )
                val_tp += tp
                val_fp += fp
                val_tn += tn
                val_fn += fn

        ema_loss_avg = val_loss_sum / max(len(val_loader), 1)
        ema_acc = ((val_tp+val_tn) / (val_tp + val_fp + val_tn + val_fn)) * 100.0
        ema_f1 = compute_f1_score(val_tp, val_fp, val_tn, val_fn)
        print(f"[EMA] Epoch {epoch+1} - Loss: {ema_loss_avg:.4f}, Acc: {ema_acc:.2f}%, F1: {ema_f1:.4f}")
        if use_wandb:
            wandb.log({"EMA/Avg loss": ema_loss_avg, "EMA/Avg acc": ema_acc, "EMA/Avg f1": ema_f1, "epoch": epoch + 1})

        del val_anchors, val_candidates, val_emb_anchor, val_emb_candidate, val_loss
        # -----------------------------
        # Early Stopping & Checkpoint
        # -----------------------------
        ckpt_name = f"epoch_{epoch+1}.pth.tar"
        ckpt_path = os.path.join(save_dir, ckpt_name)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, ckpt_path)
        print(f"=> Checkpoint saved: {ckpt_path}")
        
        ema_ckpt_name = f"epoch_{epoch+1}_ema.pth.tar"
        ema_ckpt_path = os.path.join(save_dir, ema_ckpt_name)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': ema_model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, ckpt_path)
        print(f"=> EMA Checkpoint saved: {ema_ckpt_path}")

       #if val_acc > best_val_acc:
       #    best_val_acc = val_acc
       #    epochs_no_improve = 0
       #else:
       #    epochs_no_improve += 1
       #
       #if epochs_no_improve >= patience:
       #    print(f"Early stopping triggered at epoch {epoch+1}")
       #    break

        torch.cuda.empty_cache()
    
    print("===== Training Complete =====")


if __name__ == '__main__':
    main()
