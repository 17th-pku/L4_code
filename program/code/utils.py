import json
import os
from typing import Dict, List, Tuple
import csv

import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile

# Allow loading truncated/corrupted images without raising OSError
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageCSVClassificationDataset(Dataset):
    """A simple dataset that reads image paths and labels from CSV.

    CSV format: img_name,label
    Images are expected under a root directory.
    """

    def __init__(self, root_dir: str, csv_path: str, transform=None, label_map: Dict[str, int] = None):
        self.root_dir = root_dir
        self.transform = transform
        self.items: List[Tuple[str, int]] = []

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = [row for row in reader if row]
            # Detect header
            if rows and rows[0] and str(rows[0][0]).lower() in ("img_name", "image", "filename"):
                rows = rows[1:]
            for row in rows:
                if len(row) < 2:
                    continue
                name, label = row[0].strip(), str(row[1]).strip()
                if label_map is not None:
                    label_id = label_map.get(label, int(label))
                else:
                    label_id = int(label)
                # Resolve to an existing path; skip if not found
                if os.path.isabs(name) and os.path.isfile(name):
                    resolved = name
                else:
                    base = os.path.basename(name)
                    candidates = [
                        os.path.join(self.root_dir, name),
                        os.path.join(self.root_dir, base),
                        os.path.join(self.root_dir, "images", base),
                    ]
                    resolved = None
                    for p in candidates:
                        if os.path.isfile(p):
                            resolved = p
                            break
                if resolved is not None:
                    self.items.append((resolved, label_id))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        image_path, label = self.items[idx]
        img = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def save_label_map(path: str, label_map: Dict[str, int]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)


def load_label_map(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_label_map_from_csv(csv_path: str) -> Dict[str, int]:
    """Read CSV and build a stable mapping from original label ids to [0..N-1].

    Treat labels as strings to be robust to non-sequential integers.
    """
    labels = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = [row for row in reader if row]
        if rows and rows[0] and str(rows[0][0]).lower() in ("img_name", "image", "filename"):
            rows = rows[1:]
        for row in rows:
            if len(row) < 2:
                continue
            labels.append(str(row[1]).strip())
    uniq = sorted(set(labels), key=lambda x: (len(x), x))
    return {lab: i for i, lab in enumerate(uniq)}


def invert_label_map(label_map: Dict[str, int]) -> Dict[int, str]:
    return {v: k for k, v in label_map.items()}