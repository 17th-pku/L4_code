import os
import time
import json
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

try:
    import timm
    from timm.data import create_transform
except Exception:
    timm = None

from model import build_model, get_data_config, save_config
from utils import ImageCSVClassificationDataset, build_label_map_from_csv


@dataclass
class TrainConfig:
    backbone: str = "tf_efficientnetv2_l_in21ft1k"
    input_size: int = 512
    num_classes: int = 100
    epochs_stage1: int = 15
    epochs_stage2: int = 20
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 0.05
    mixup_prob: float = 0.2
    cutmix_prob: float = 0.2
    label_smoothing: float = 0.05
    num_workers: int = 4
    amp: bool = True
    output_dir: str = "/root/autodl-tmp/program/model"
    train_root: str = "/root/autodl-tmp/data/train"
    train_csv: str = "/root/autodl-tmp/data/train/train_labels.csv"
    config_json: str = "/root/autodl-tmp/program/model/config.json"


def create_dataloaders(cfg: TrainConfig, data_cfg, label_map):
    if timm is None:
        raise RuntimeError("timm is required but not installed.")
    transform_train = create_transform(
        input_size=(3, cfg.input_size, cfg.input_size),
        is_training=True,
        auto_augment="rand-m9-mstd0.5-inc1",
        interpolation=data_cfg.get("interpolation", "bicubic"),
        mean=data_cfg.get("mean"),
        std=data_cfg.get("std"),
    )
    ds_train = ImageCSVClassificationDataset(cfg.train_root, cfg.train_csv, transform=transform_train, label_map=label_map)
    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    return dl_train


def train_one_stage(model, loader, cfg: TrainConfig, device, epochs: int):
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_correct = 0
        total = 0
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.amp):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total += imgs.size(0)

        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs} - loss: {total_loss/total:.4f} acc: {total_correct/total:.4f}")


def main():
    cfg = TrainConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build label mapping from CSV to handle non-sequential IDs
    label_map = build_label_map_from_csv(cfg.train_csv)
    num_classes = len(label_map)

    data_cfg = get_data_config(cfg.backbone, cfg.input_size)
    dl_train = create_dataloaders(cfg, data_cfg, label_map)

    # Prefer local pretrained weights if available to comply with submission rules
    local_w_path = os.path.join("/root/autodl-tmp/model", f"{cfg.backbone}.pth")
    use_local = os.path.isfile(local_w_path)
    model = build_model(
        cfg.backbone,
        num_classes,
        pretrained=not use_local,
        pretrained_path=local_w_path if use_local else None,
        drop_rate=0.2,
        load_strict=False,
    )
    model.to(device)

    # Stage 1: freeze most layers for quick adaptation
    for p in model.parameters():
        p.requires_grad = False
    if hasattr(model, 'get_classifier'):
        head = model.get_classifier()
        for p in head.parameters():
            p.requires_grad = True

    train_one_stage(model, dl_train, cfg, device, epochs=cfg.epochs_stage1)

    # Stage 2: unfreeze and fine-tune end-to-end
    for p in model.parameters():
        p.requires_grad = True
    train_one_stage(model, dl_train, cfg, device, epochs=cfg.epochs_stage2)

    # Save checkpoint and config
    ckpt_path = os.path.join(cfg.output_dir, "best_model.pth")
    torch.save(model.state_dict(), ckpt_path)

    save_config(cfg.config_json, {
        "backbone": cfg.backbone,
        "num_classes": num_classes,
        "input_size": cfg.input_size,
        "drop_rate": 0.2,
        "mean": data_cfg.get("mean"),
        "std": data_cfg.get("std"),
        "interpolation": data_cfg.get("interpolation"),
        # Embed label_map directly to avoid extra files in submission package
        "label_map": label_map,
    })
    print(f"Saved checkpoint to {ckpt_path}")
    print(f"Saved config to {cfg.config_json}")


if __name__ == "__main__":
    main()