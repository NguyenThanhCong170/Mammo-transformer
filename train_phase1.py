import torch
from torch.amp import autocast, GradScaler

from data.dataset import build_dataloaders
from models.mammo_transformer import SwinV2Backbone
from models.projection_head import ProjectionHead
from utils.Supcon_loss import MultiViewNTXentLoss
from configs.config import Config
cfg = Config()

# 1. Bật GradScaler cho AMP
scaler = GradScaler()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = GradScaler(device="cuda" if device.type == "cuda" else "cpu")
model = SwinV2Backbone(
        model_name=cfg.model.backbone_name,
        pretrained=cfg.model.pretrained_phase1,
        img_width = cfg.data.image_size[0],
        img_height = cfg.data.image_size[1]
    ).to(device)
model.train()


projection_head = ProjectionHead().to(device)
projection_head.train()


multi_view_criterion = MultiViewNTXentLoss()

optimizer = torch.optim.AdamW(
    list(filter(lambda p: p.requires_grad, model.parameters())) + 
    list(projection_head.parameters()), 
    lr = cfg.train.lr_phase1
)

train_loader, val_loader, test_loader = build_dataloaders(
        csv_path=cfg.data.csv_path,
        image_size=cfg.model.backbone_img_size,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.data.num_workers,
        val_ratio=cfg.data.val_ratio,
        seed=cfg.data.seed,
        aug_level=cfg.data.aug_level,
    )
VIEW_KEYS = ["L_MLO", "L_CC", "R_MLO", "R_CC"]
# Trong vòng lặp train:
for batch_idx, batch in enumerate(train_loader):
    # images shape: [8, 4, C, H, W]
    images_dict = batch["images"]
    # Stack 4 view theo đúng thứ tự VIEW_KEYS -> [B, num_views=4, C, H, W]
    images = torch.stack([images_dict[k] for k in VIEW_KEYS], dim=1)
    B, num_views, C, H, W = images.shape
    # Gộp lại thành [32, C, H, W] để đưa vào SwinV2
    images = images.view(-1, C, H, W).to(device, non_blocking=True)
    optimizer.zero_grad()

    # 2. Dùng autocast cho FP16
    with autocast():
        # Lấy features từ SwinV2 
        # (Không lo OOM vì đã bật Checkpointing và Freeze)
        features = model.forward(images) # Tùy hàm của thư viện, có thể trả về [32, 1024]
        
        # Đưa qua Projection Head
        z = projection_head(features) # [32, 128]
        
        # Đưa trở lại shape [B, num_views, 128] cho hàm Multi-view Loss
        z = z.view(B, num_views, -1)
        
        # Tính SupCon Loss
        loss = multi_view_criterion(z)

    # 3. Backward và Update bằng Scaler
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    print(f"Batch {batch_idx} | Loss: {loss.item():.4f}")