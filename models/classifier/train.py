"""
train.py
========
Training, temperature calibration, and ONNX export for the Stage 3 MobileNetV3
pathology classifier (CNV / DME / DRUSEN / NORMAL) on 64x64 grayscale crops.

Data: patient-level splits from data/splits/{train,val,test}.csv (see data_scripts/make_splits.py).
Images are resolved from the CSV path, falling back to data/raw/kermany/{class}/{filename}
so splits created on another machine/path still work. Training NEVER falls back to synthetic data.

Crops: random square crops (side 64-160 px of the 256x256 scan) resized to 64x64 for training;
a fixed centre crop (side 128) for validation/test. Labels are the scan-level diagnosis, so
crops are a proxy for the lesion ROI crops produced at runtime by ScanSession.

Outputs:
    exports/pathology_classifier_best.pt   (state dict incl. fitted temperature)
    exports/pathology_classifier.onnx      (temperature-scaled class probabilities)
"""

import os
import sys
import csv
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src_python"))
from onnx_model_io import export_pytorch_to_onnx, verify_pytorch_onnx_parity
from pathology import PATHOLOGY_CLASSES, ROI_CROP_SIZE

try:
    from mobilenet_v3 import CalibratedClassifier, MobileNetV3Classifier, fit_temperature
except ImportError:
    from models.classifier.mobilenet_v3 import CalibratedClassifier, MobileNetV3Classifier, fit_temperature

SCAN_SIZE = 256
CLASS_TO_INDEX = {name: i for i, name in enumerate(PATHOLOGY_CLASSES)}


def resolve_image_path(row: dict, raw_root: Path) -> Optional[Path]:
    """Finds a split row's image: the CSV path, else raw_root/{class}/{filename}."""
    csv_path = Path(row["image_path"])
    if csv_path.exists():
        return csv_path
    fallback = raw_root / row["class"] / csv_path.name
    return fallback if fallback.exists() else None


def load_split(split_csv: str, raw_root: Path) -> Tuple[List[np.ndarray], List[int]]:
    """Loads a split into memory as 256x256 float32 scans and integer labels."""
    images: List[np.ndarray] = []
    labels: List[int] = []
    missing = 0
    with open(split_csv, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            cls = row["class"].strip().upper()
            if cls not in CLASS_TO_INDEX:
                continue
            path = resolve_image_path({**row, "class": cls}, raw_root)
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if path else None
            if img is None:
                missing += 1
                continue
            img = cv2.resize(img, (SCAN_SIZE, SCAN_SIZE), interpolation=cv2.INTER_AREA)
            images.append(img.astype(np.float32) / 255.0)
            labels.append(CLASS_TO_INDEX[cls])

    if not images:
        raise RuntimeError(
            f"No images could be loaded for {split_csv} (missing={missing}). "
            f"Check data/raw/kermany or the image_path column."
        )
    if missing:
        print(f"[!] {split_csv}: {missing} images missing; loaded {len(images)}")
    return images, labels


class CropDataset(Dataset):
    """64x64 crops from in-memory scans; random crops when train=True, else centre crop."""

    def __init__(self, images: List[np.ndarray], labels: List[int], train: bool, seed: int = 42):
        self.images = images
        self.labels = labels
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.images)

    def _crop(self, img: np.ndarray) -> np.ndarray:
        if self.train:
            side = int(self.rng.integers(ROI_CROP_SIZE, 161))
            y = int(self.rng.integers(0, SCAN_SIZE - side + 1))
            x = int(self.rng.integers(0, SCAN_SIZE - side + 1))
        else:
            side = 128
            y = x = (SCAN_SIZE - side) // 2
        crop = img[y:y + side, x:x + side]
        crop = cv2.resize(crop, (ROI_CROP_SIZE, ROI_CROP_SIZE), interpolation=cv2.INTER_AREA)
        if self.train and self.rng.random() < 0.5:
            crop = crop[:, ::-1]
        return np.ascontiguousarray(crop, dtype=np.float32)

    def __getitem__(self, idx: int):
        crop = torch.from_numpy(self._crop(self.images[idx])[np.newaxis])
        return crop, self.labels[idx]


@torch.no_grad()
def collect_logits(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_logits, all_labels = [], []
    for x, y in loader:
        all_logits.append(model(x.to(device)).cpu())
        all_labels.append(y)
    return torch.cat(all_logits), torch.cat(all_labels).long()


def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, bins: int = 10) -> float:
    conf, pred = probs.max(dim=1)
    correct = (pred == labels).float()
    edges = torch.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.float().mean().item() * abs(correct[mask].mean().item() - conf[mask].mean().item())
    return float(ece)


def pick_device(force_cpu: bool) -> torch.device:
    if not force_cpu and torch.backends.mps.is_available():
        return torch.device("mps")
    if not force_cpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_classifier(args) -> int:
    torch.manual_seed(args.seed)
    device = pick_device(args.force_cpu)
    raw_root = Path(args.raw_root)
    print(f"[*] Device: {device}")

    train_imgs, train_lbls = load_split(os.path.join(args.splits_dir, "train.csv"), raw_root)
    val_imgs, val_lbls = load_split(os.path.join(args.splits_dir, "val.csv"), raw_root)
    print(f"[+] Loaded train={len(train_imgs)} val={len(val_imgs)} scans")

    train_loader = DataLoader(CropDataset(train_imgs, train_lbls, True, args.seed), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(CropDataset(val_imgs, val_lbls, False), batch_size=args.batch_size)

    model = MobileNetV3Classifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_acc, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)

        logits, labels = collect_logits(model, val_loader, device)
        val_acc = (logits.argmax(1) == labels).float().mean().item()
        print(f"    Epoch [{epoch}/{args.epochs}] train_loss={running / len(train_imgs):.4f} val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.to(device)

    # --- Temperature scaling on held-out validation logits ---
    logits, labels = collect_logits(model, val_loader, device)
    ece_before = expected_calibration_error(F.softmax(logits, dim=1), labels)
    temperature = fit_temperature(logits, labels)
    model.temperature.fill_(temperature)
    ece_after = expected_calibration_error(F.softmax(logits / temperature, dim=1), labels)
    print(f"[+] Best val_acc={best_acc:.4f} | T={temperature:.3f} | ECE {ece_before:.4f} -> {ece_after:.4f} (on val)")

    # --- Optional test-split report ---
    test_csv = os.path.join(args.splits_dir, "test.csv")
    if os.path.exists(test_csv):
        t_imgs, t_lbls = load_split(test_csv, raw_root)
        t_logits, t_labels = collect_logits(
            model, DataLoader(CropDataset(t_imgs, t_lbls, False), batch_size=args.batch_size), device
        )
        print(f"[+] Test accuracy: {(t_logits.argmax(1) == t_labels).float().mean().item():.4f} ({len(t_imgs)} scans)")

    # --- Save checkpoint + export ONNX with parity check ---
    model.to("cpu").eval()
    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.checkpoint)
    print(f"[+] Saved checkpoint: {args.checkpoint}")

    export_pytorch_to_onnx(
        CalibratedClassifier(model),
        args.output,
        height=ROI_CROP_SIZE,
        width=ROI_CROP_SIZE,
        input_name="roi_crop",
        output_name="class_probabilities",
        dynamic_spatial=False,
    )
    ok, max_diff, _ = verify_pytorch_onnx_parity(
        CalibratedClassifier(model), args.output, height=ROI_CROP_SIZE, width=ROI_CROP_SIZE
    )
    print(f"[+] Exported {args.output} ({os.path.getsize(args.output) / 1024:.1f} KB) | parity ok={ok} max_diff={max_diff:.2e}")

    # Maintain backward-compatible copy for legacy consumers
    legacy_path = os.path.join(os.path.dirname(args.output) or ".", "pathology_classifier.onnx")
    if os.path.exists(args.output) and os.path.abspath(args.output) != os.path.abspath(legacy_path):
        import shutil
        shutil.copyfile(args.output, legacy_path)

    return 0 if ok else 1


def parse_args():
    p = argparse.ArgumentParser(description="Train Stage 3 MobileNetV3 pathology classifier")
    p.add_argument("--splits-dir", type=str, default="data/splits")
    p.add_argument("--raw-root", type=str, default="data/raw/kermany")
    p.add_argument("--checkpoint", type=str, default="exports/pathology_classifier_best.pt")
    p.add_argument("--output", type=str, default="exports/mobilenet_v3_pathology_classifier.onnx")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force-cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(train_classifier(parse_args()))
