"""
models/despeckler/train_n2n.py
==============================
Neighbor2Neighbor (N2N) self-supervised training pipeline for Stage 1 ConvUNet Despeckler.

Pre-registered in EXPERIMENT_PLAN.md:
- unet_n2n: Neighbor2Neighbor ConvUNet trained on adjacent / neighbor B-scans.
- Same architecture as unet_synth: ConvUNet with continuous SiLU activations.
- Self-supervised: requires ZERO ground-truth clean images.

Outputs:
    exports/despeckler_n2n_best.pt
    exports/unet_n2n.onnx
"""

import os
import sys
import glob
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models.despeckler.model import ConvUNet
from src_python.onnx_model_io import export_pytorch_to_onnx, verify_pytorch_onnx_parity


def generate_subpixel_neighbor_pair(img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sub-pixel Neighbor2Neighbor sampling (Huang et al., CVPR 2021).
    Given input scan [B, 1, H, W], extracts two downsampled sub-images [B, 1, H/2, W/2]
    sampled from adjacent diagonal pixel pairs within each 2x2 neighborhood.
    """
    sub1 = img[:, :, 0::2, 0::2]
    sub2 = img[:, :, 1::2, 1::2]
    return sub1, sub2


class N2NDataset(Dataset):
    """
    Dataset for Neighbor2Neighbor training.
    Supports:
    1. Paired adjacent B-scans from volume directories (e.g. Duke SD-OCT).
    2. Single noisy clinical scans where sub-pixel neighbor pairs are formed on the fly.
    """

    def __init__(
        self,
        duke_dir: str = "data/raw/duke/Sparsity_SDOCT_DATASET_2012",
        kermany_csv: str = "data/splits/train.csv",
        raw_root: str = "data/raw/kermany",
        scan_size: int = 256,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ):
        self.scan_size = scan_size
        self.rng = np.random.default_rng(seed)
        self.samples: List[Tuple[str, str]] = []  # (path1, path2)

        # 1. Collect paired adjacent scans from Duke 2012
        duke_path = Path(duke_dir) if duke_dir else None
        if duke_path and duke_path.is_dir():
            for subj_dir in sorted(duke_path.iterdir()):
                if not subj_dir.is_dir():
                    continue
                raw_files = list(subj_dir.rglob("*Raw Image.tif"))
                if len(raw_files) >= 2:
                    for i in range(len(raw_files) - 1):
                        self.samples.append((str(raw_files[i]), str(raw_files[i + 1])))

        # 2. Add single clinical scans for sub-pixel N2N
        kermany_path = Path(kermany_csv) if kermany_csv else None
        if kermany_path and kermany_path.is_file():
            import csv
            with open(kermany_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    img_p = Path(row["image_path"])
                    if not img_p.exists():
                        img_p = Path(raw_root) / row["class"] / img_p.name
                    if img_p.exists():
                        self.samples.append((str(img_p), str(img_p)))

        if max_samples is not None and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]

        if not self.samples:
            print("[!] Warning: No images found for N2N training dataset.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        p1, p2 = self.samples[idx]
        img1 = cv2.imread(p1, cv2.IMREAD_GRAYSCALE)
        if img1 is None:
            img1 = np.zeros((self.scan_size, self.scan_size), dtype=np.uint8)
        img1 = cv2.resize(img1, (self.scan_size, self.scan_size), interpolation=cv2.INTER_AREA)
        f1 = img1.astype(np.float32) / 255.0

        if p1 != p2:
            img2 = cv2.imread(p2, cv2.IMREAD_GRAYSCALE)
            if img2 is None:
                img2 = img1
            img2 = cv2.resize(img2, (self.scan_size, self.scan_size), interpolation=cv2.INTER_AREA)
            f2 = img2.astype(np.float32) / 255.0
        else:
            f2 = f1

        t1 = torch.from_numpy(f1[np.newaxis])
        t2 = torch.from_numpy(f2[np.newaxis])
        return t1, t2


def train_n2n(
    output_dir: str = "exports",
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-3,
    gamma: float = 2.0,
    seed: int = 42,
    max_samples: Optional[int] = None,
    force_cpu: bool = False,
) -> str:
    """Trains ConvUNet with self-supervised Neighbor2Neighbor loss."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(
        "cpu" if force_cpu else ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    )
    print(f"[*] Training unet_n2n on device: {device}")

    dataset = N2NDataset(max_samples=max_samples, seed=seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"[+] Loaded {len(dataset)} N2N training samples")

    model = ConvUNet(in_channels=1, base_ch=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    os.makedirs(output_dir, exist_ok=True)
    best_ckpt = os.path.join(output_dir, "despeckler_n2n_best.pt")
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for x1, x2 in loader:
            x1, x2 = x1.to(device), x2.to(device)
            optimizer.zero_grad()

            if torch.equal(x1, x2):
                # Sub-pixel neighbor sampling on identical scan
                sub1, sub2 = generate_subpixel_neighbor_pair(x1)
                pred1 = model(sub1)
                pred2 = model(sub2)
                loss_recon = F.mse_loss(pred1, sub2) + F.mse_loss(pred2, sub1)
                # Consistency regularization (Huang et al. 2021)
                diff1 = pred1 - sub1
                diff2 = pred2 - sub2
                loss_reg = F.mse_loss(diff1, diff2)
                loss = loss_recon + gamma * loss_reg
            else:
                # Paired adjacent B-scans
                pred1 = model(x1)
                pred2 = model(x2)
                loss = F.mse_loss(pred1, x2) + F.mse_loss(pred2, x1)

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x1.size(0)

        epoch_loss = running_loss / max(1, len(dataset))
        print(f"    Epoch [{epoch}/{epochs}] N2N Loss: {epoch_loss:.5f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), best_ckpt)

    print(f"[+] unet_n2n training complete! Best checkpoint: {best_ckpt}")

    # Export to ONNX with verified parity
    onnx_path = os.path.join(output_dir, "unet_n2n.onnx")
    model.load_state_dict(torch.load(best_ckpt, map_location="cpu"))
    model.eval()

    export_pytorch_to_onnx(
        model,
        onnx_path,
        height=256,
        width=256,
        input_name="raw_frame",
        output_name="despeckled_frame",
        dynamic_spatial=True,
    )
    ok, max_diff, _ = verify_pytorch_onnx_parity(model, onnx_path, height=256, width=256)
    print(f"[+] Exported {onnx_path} | ONNX parity ok={ok} max_diff={max_diff:.2e}")

    return best_ckpt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train unet_n2n Neighbor2Neighbor ConvUNet")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=str, default="exports")
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()

    train_n2n(
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        force_cpu=args.force_cpu,
    )
