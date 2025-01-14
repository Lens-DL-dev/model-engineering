import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast  # 수정됨
from tqdm import tqdm

# 내부 모듈
from utils.config import load_config
from utils.dataset import ContrastiveFashionDataset, collate_fn_contrastive
from utils.losses import ContrastiveLoss
from utils.metrics import calculate_accuracy_contrastive
from utils.visualization import save_contrastive_matrix, save_topk_image_samples
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', type=str)
    args = parser.parse_args()

    # -----------------------------
    # 1. Load Config
    # -----------------------------
    config = load_config(args.config)
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    lr = config['training']['learning_rate']
    num_workers = config['training']['num_workers']
    margin = config['loss'].get('margin', 1.0)
    distance_metric = config['loss'].get('distance_metric', 'euclidean')
    save_dir = config['training'].get('save_dir', './checkpoints')
    os.makedirs(save_dir, exist_ok=True)

    grad_acc_steps = config['training'].get('gradient_accumulation_steps', 1)

    # early stopping
    patience = config['training'].get('early_stopping_patience', 5)
    best_val_acc = 0.0
    epochs_no_improve = 0

    # checkpoint
    checkpoint_path = config['model']['checkpoint']  
    resume = config['model'].get('resume', False)
    image_size = config['model'].get('image_size', 224)
    embed_dim = config['model'].get('embed_dim', 512)

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
    data_root = config['data']['root']
    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')

    negative_count = config['dataset'].get('negative_count', 4)

    train_dataset = ContrastiveFashionDataset(
        root_dir=train_dir,
        wearing_info_path=os.path.join(train_dir, 'wearing_info.json'),
        is_train=True,
        image_size=image_size,
        negative_count=negative_count
    )
    val_dataset = ContrastiveFashionDataset(
        root_dir=val_dir,
        wearing_info_path=os.path.join(val_dir, 'wearing_info.json'),
        is_train=False,
        image_size=image_size,
        negative_count=negative_count
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
    model = EfficientNetV2L(pretrained=True, embed_dim=embed_dim).to(device)

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

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        
        # (개선) 에폭 전체 정확도 계산 위해
        epoch_correct = 0
        epoch_total = 0

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
        for i, (anchors, candidates, labels) in pbar:
            global_step += 1

            anchors = anchors.to(device, non_blocking=True)
            candidates = candidates.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # AMP
            with autocast(device_type='cuda', dtype=torch.float16):
                emb_anchor = model(anchors)
                emb_candidate = model(candidates)
                loss = criterion(emb_anchor, emb_candidate, labels)
                loss = loss / grad_acc_steps

            scaler.scale(loss).backward()

            if (i + 1) % grad_acc_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * grad_acc_steps

            # (개선) 배치 내 correct/total
            with torch.no_grad():
                # 배치별 Accuracy
                dist_acc = calculate_accuracy_contrastive(
                    emb_anchor, emb_candidate, labels,
                    threshold=margin,
                    distance_metric=distance_metric
                )
                # dist_acc는 % 단위
                # 배치 정답 개수
                batch_correct = dist_acc / 100.0 * labels.size(0)
                epoch_correct += batch_correct
                epoch_total += labels.size(0)

            avg_loss = running_loss / (i + 1)
            # 누적 correct / total -> 에폭 누적 평균
            current_epoch_acc = 0.0
            if epoch_total > 0:
                current_epoch_acc = (epoch_correct / epoch_total) * 100.0

            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "acc(%)": f"{current_epoch_acc:.2f}"
            })
            
            # -----------------------------
            # (NEW) 100번째 배치마다 시각화
            # -----------------------------
            if (i % 100 == 0) and (i > 0):
                vis_dir = os.path.join("vis_results", f"epoch_{epoch+1}_step_{i}")
                os.makedirs(vis_dir, exist_ok=True)

                # 1) Contrastive Distance Plot
                save_contrastive_matrix(
                    emb_anchor, emb_candidate, labels,
                    save_path=os.path.join(vis_dir, "dist_plot.png"),
                    distance_metric=distance_metric,
                    margin=margin
                )
                # 2) top-k 이미지 samples
                save_topk_image_samples(
                    anchors, candidates,
                    emb_anchor, emb_candidate,
                    k=3,
                    distance_metric=distance_metric,
                    out_dir=vis_dir
                )

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = (epoch_correct / epoch_total) * 100.0 if epoch_total > 0 else 0.0
        print(f"[Train] Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")

        # -----------------------------
        # Validation
        # -----------------------------
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_sum = 0.0

        with torch.no_grad():
            for val_i, (val_anchors, val_candidates, val_labels) in enumerate(val_loader):
                val_anchors = val_anchors.to(device)
                val_candidates = val_candidates.to(device)
                val_labels = val_labels.to(device)

                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    val_emb_anchor = model(val_anchors)
                    val_emb_candidate = model(val_candidates)
                    val_loss = criterion(val_emb_anchor, val_emb_candidate, val_labels)
                    val_loss_sum += val_loss.item()

                val_acc_batch = calculate_accuracy_contrastive(
                    val_emb_anchor, val_emb_candidate, val_labels,
                    threshold=margin,
                    distance_metric=distance_metric
                )
                # val_acc_batch는 %, 배치 정답 개수
                batch_correct = val_acc_batch / 100.0 * val_labels.size(0)
                val_correct += batch_correct
                val_total += val_labels.size(0)

        val_loss_avg = val_loss_sum / max(len(val_loader), 1)
        val_acc = (val_correct / val_total) * 100.0 if val_total > 0 else 0.0
        print(f"[Val] Epoch {epoch+1} - Loss: {val_loss_avg:.4f}, Acc: {val_acc:.2f}%")

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

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

        torch.cuda.empty_cache()

    print("===== Training Complete =====")


if __name__ == '__main__':
    main()
