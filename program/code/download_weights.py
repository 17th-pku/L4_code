"""
Helper script to download timm pretrained weights to local directory.

The competition disallows uploading pretrained weights in the submission,
so we keep them under /root/autodl-tmp/model for local training/inference.
"""

import os
import argparse
import timm
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="tf_efficientnetv2_l_in21ft1k")
    parser.add_argument("--out_dir", type=str, default="/root/autodl-tmp/model")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Preparing pretrained weights for {args.backbone}")

    # Create model with pretrained=True to trigger weight fetch into timm cache
    m = timm.create_model(args.backbone, pretrained=True, num_classes=1000)
    sd = m.state_dict()
    out_path = os.path.join(args.out_dir, f"{args.backbone}.pth")
    torch.save(sd, out_path)
    print(f"Saved weights to {out_path}")


if __name__ == "__main__":
    main()