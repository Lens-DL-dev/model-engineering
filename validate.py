import os
import torch
import argparse
from tqdm import tqdm
import wandb
from timm.utils import ModelEmaV2

# 내부 모듈
from utils.config import load_config
from utils.dataset import ContrastiveFashionDataset, collate_fn_supcon
from utils.metrics import compute_topk_accuracy
from models.convnext import TwoTowerModel

def load_checkpoint(model, filename):
    if os.path.isfile(filename):
        print(f"=> Loading checkpoint from '{filename}'")
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"=> Loaded checkpoint from '{filename}'")
    else:
        raise FileNotFoundError(f"Checkpoint not found at '{filename}'")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', type=str)
    parser.add_argument('--checkpoint', required=True, type=str, help='Path to best_model checkpoint')
    args = parser.parse_args()
    
    # Load Config
    config = load_config(args.config)
    
    # WandB 설정
    wandb_config = config.get('wandb', {})
    use_wandb = wandb_config.get('enabled', config.get('use_wandb', False))
    if use_wandb:
        wandb.init(
            project=wandb_config.get('project_name', "fashion-supcon-a100"),
            tags=wandb_config.get('tags', ["a100", "supcon", "memory-bank"]),
            config=config,
            name=wandb_config.get('run_name', "validation_run")
        )
    
    # Device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 모델 설정
    model = TwoTowerModel(
        backbone_wear=config['model'].get('backbone_wear', 'convnext_tiny'),
        backbone_prod=config['model'].get('backbone_prod', 'convnext_tiny'),
        embed_dim=config['model'].get('embed_dim', 512),
        use_timm=config['model'].get('use_timm', False)
    ).to(device)
    
    # EMA 적용
    ema = ModelEmaV2(model, decay=0.999)
    
    # Checkpoint 로드
    load_checkpoint(model, args.checkpoint)
    if os.path.exists(args.checkpoint.replace("best_model.pth.tar", "best_model_ema.pth.tar")):
        load_checkpoint(ema.module, args.checkpoint.replace("best_model.pth.tar", "best_model_ema.pth.tar"))
    
    # Dataset / Dataloader 설정
    val_dataset = ContrastiveFashionDataset(
        root_dir=config['data']['val_dir'],
        metainfo_path=config['data']['val_metainfo_path'],
        is_train=False,
        image_size=config['model'].get('image_size', 384),
        n_mask_channels=config['data'].get('n_mask_channels', 0),
        mode='supcon',
        config=config
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        drop_last=False,
        collate_fn=collate_fn_supcon,
        pin_memory=True
    )
    
    # 모델 평가
    def evaluate(model, name="model"):
        model.eval()
        all_wearing_emb = []
        all_product_emb = []
        all_product_codes = []
        
        with torch.no_grad():
            for val_wearing, val_product, val_codes in tqdm(val_loader, desc=f"[Validation] {name}"):
                val_wearing = val_wearing.to(device)
                val_product = val_product.to(device)
                
                emb_wearing, emb_product = model(val_wearing, val_product)
                all_wearing_emb.append(emb_wearing)
                all_product_emb.append(emb_product)
                all_product_codes.extend(val_codes)
        
        query_emb = torch.cat(all_wearing_emb, dim=0)
        candidate_emb = torch.cat(all_product_emb, dim=0)
        topk_acc = compute_topk_accuracy(query_emb, candidate_emb, topk=(1,5,10))
        print(f"[{name}] Top1: {topk_acc[1]:.2f}%, Top5: {topk_acc[5]:.2f}%, Top10: {topk_acc[10]:.2f}%")
        
        if use_wandb:
            wandb.log({
                f"val/{name}_top1": topk_acc[1],
                f"val/{name}_top5": topk_acc[5],
                f"val/{name}_top10": topk_acc[10]
            })
    
    # 기본 모델 평가
    evaluate(model, name="model")
    
    # EMA 모델 평가
    evaluate(ema.module, name="ema")
    
    if use_wandb:
        wandb.finish()

if __name__ == '__main__':
    main()
