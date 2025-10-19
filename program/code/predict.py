import os
import sys
import argparse
import csv
import time
from typing import List

import torch
from PIL import Image, ImageFile

try:
    import timm
    from timm.data import create_transform
except Exception:
    timm = None

from model import load_model_for_inference, load_config
from utils import invert_label_map


def list_images(root_dir: str) -> List[str]:
    names = []
    for n in os.listdir(root_dir):
        if n.lower().endswith((".jpg", ".jpeg", ".png")):
            names.append(n)
    names.sort()
    return names


def main():
    parser = argparse.ArgumentParser(description="Run inference and save submission.csv")
    parser.add_argument("test_dir", nargs="?", default=None, help="Path to test images directory")
    args = parser.parse_args()

    model_dir = "/root/autodl-tmp/program/model"
    config_path = os.path.join(model_dir, "config.json")
    ckpt_path = os.path.join(model_dir, "best_model.pth")
    # Prefer /test/images, fallback to /test if /images doesn't exist
    test_dir_default = "/root/autodl-tmp/data/test/images"
    if args.test_dir and os.path.isdir(args.test_dir):
        test_dir = args.test_dir
    else:
        if os.path.isdir(test_dir_default):
            test_dir = test_dir_default
        else:
            alt = "/root/autodl-tmp/data/test"
            test_dir = alt if os.path.isdir(alt) else test_dir_default
    out_csv = "/root/autodl-tmp/program/results/submission.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_for_inference(config_path, ckpt_path, device)

    input_size = int(cfg.get("input_size", 512))
    mean = cfg.get("mean", [0.485, 0.456, 0.406])
    std = cfg.get("std", [0.229, 0.224, 0.225])
    interpolation = cfg.get("interpolation", "bicubic")

    transform = None
    if timm is not None:
        transform = create_transform(
            input_size=(3, input_size, input_size),
            is_training=False,
            interpolation=interpolation,
            mean=mean,
            std=std,
        )

    names = list_images(test_dir)

    # Map internal class ids back to original IDs for submission
    inv_map = None
    lm = cfg.get("label_map")
    if isinstance(lm, dict) and lm:
        inv_map = invert_label_map(lm)
    rows = []
    times = []
    for n in names:
        path = os.path.join(test_dir, n)
        img = Image.open(path).convert("RGB")
        if transform is not None:
            x = transform(img)
        else:
            from torchvision import transforms
            tfm = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
            x = tfm(img)

        x = x.unsqueeze(0).to(device)
        t0 = time.time()
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
        dt = (time.time() - t0) * 1000
        times.append(dt)
        pred_idx = int(pred.item())
        pred_label = inv_map.get(pred_idx, str(pred_idx)) if inv_map else str(pred_idx)
        rows.append((n, pred_label, float(conf.item())))

    avg_ms = sum(times) / max(len(times), 1)
    print(f"Average inference time: {avg_ms:.2f} ms per image")
    # Speed fallback: if >100ms, advise lowering resolution or disabling TTA (here we only log)
    if avg_ms > 100:
        print("Warning: avg inference >100ms. Consider input_size=448 or disabling extra transforms.")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["img_name", "predicted_class", "confidence"])
        for r in rows:
            w.writerow(list(r))

    print(f"Saved predictions to {out_csv}")


if __name__ == "__main__":
    main()
    # Allow loading truncated/corrupted images without raising OSError
    ImageFile.LOAD_TRUNCATED_IMAGES = True