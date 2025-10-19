import json
import os
from typing import Dict, Any, Optional

import torch
import torch.nn as nn

try:
    import timm
    from timm.data import resolve_data_config
except Exception:
    timm = None


def build_model(
    backbone: str,
    num_classes: int,
    pretrained: bool = True,
    drop_rate: float = 0.2,
    pretrained_path: Optional[str] = None,
    load_strict: bool = False,
) -> nn.Module:
    """Create a classification model using timm.

    Parameters
    ----------
    backbone: model name in timm (e.g., 'tf_efficientnetv2_l_in21ft1k').
    num_classes: number of output classes.
    pretrained: whether to load ImageNet pretrained weights.
    drop_rate: classifier dropout rate if supported.
    """
    if timm is None:
        raise RuntimeError("timm is required but not installed.")

    model = timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )
    # Optional: manually load pretrained weights from local path.
    # Filter out keys whose shapes don't match (e.g., classifier for different num_classes).
    if pretrained_path is not None and os.path.isfile(pretrained_path):
        state = torch.load(pretrained_path, map_location="cpu")
        if isinstance(state, dict):
            # common keys: 'state_dict', 'model'
            if 'state_dict' in state:
                state = state['state_dict']
            elif 'model' in state:
                state = state['model']

        model_state = model.state_dict()
        filtered = {}
        mismatched = []
        for k, v in state.items():
            if k in model_state:
                if tuple(model_state[k].shape) == tuple(v.shape):
                    filtered[k] = v
                else:
                    mismatched.append(k)
        if mismatched:
            print(f"[load_pretrained] Ignored shape-mismatched keys: {mismatched[:6]}{'...' if len(mismatched)>6 else ''}")
        missing = [k for k in model_state.keys() if k not in filtered]
        if missing:
            # Only log; classifier/head is expected to be missing since num_classes differs
            print(f"[load_pretrained] Missing keys will be randomly initialized: {missing[:6]}{'...' if len(missing)>6 else ''}")
        model.load_state_dict(filtered, strict=False)
    return model


def get_data_config(backbone: str, input_size: int) -> Dict[str, Any]:
    """Resolve data config (mean/std/interpolation) for the given backbone.

    Falls back to ImageNet defaults if timm isn't available.
    """
    if timm is None:
        return {
            "input_size": (3, input_size, input_size),
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "interpolation": "bicubic",
            "crop_pct": 1.0,
        }

    # Create a lightweight instance to fetch pretrained_cfg
    tmp_model = timm.create_model(backbone, pretrained=False)
    cfg = resolve_data_config(model=tmp_model)
    cfg["input_size"] = (3, input_size, input_size)
    return cfg


def save_config(path: str, cfg: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model_for_inference(config_path: str, ckpt_path: str, device: Optional[torch.device] = None) -> nn.Module:
    """Load a trained model for inference.

    Expects config.json to contain backbone & num_classes.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_config(config_path)
    backbone = cfg.get("backbone")
    num_classes = int(cfg.get("num_classes"))
    drop_rate = float(cfg.get("drop_rate", 0.2))

    model = build_model(backbone=backbone, num_classes=num_classes, pretrained=False, drop_rate=drop_rate)
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model