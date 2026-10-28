"""
train_autoencoder.py
====================
Training and ONNX export pipeline for the Stage 2 ConvAutoencoder tissue baseline.
Learns healthy normal tissue representation and exports to ONNX for the C++ AnomalyCalculus engine.

Supports both:
1. Real clinical negative baseline scans (extracted via data_scripts/negative_mining.py)
2. Synthetic tissue generator fallback (for quick offline / CI testing)
"""

import os
import sys
import csv
import argparse
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src_python"))
from onnx_model_io import export_pytorch_to_onnx, verify_pytorch_onnx_parity

try:
    from autoencoder import ConvAutoencoder
except ImportError:
    from models.anomaly_detector.autoencoder import ConvAutoencoder


class HealthyTissueDataset(Dataset):
    """
    Dataset strictly loading healthy/normal tissue scans for negative-only baseline learning.
    If negative split CSV is missing or empty, falls back gracefully to synthetic tissue generation.
    """
    def __init__(
        self,
        split_csv: Optional[str] = "data/splits/negative_train.csv",
        height: int = 128,
        width: int = 128,
        num_synthetic_fallback: int = 200,
        seed: int = 42,
    ):
        self.height = height
        self.width = width
        self.rng = np.random.default_rng(seed)
        self.file_paths: List[str] = []

        if split_csv and os.path.exists(split_csv):
            with open(split_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    p = row.get("primary_path") or row.get("processed_path") or row.get("raw_path")
                    if p and os.path.exists(p):
                        self.file_paths.append(p)

        self.synthetic = len(self.file_paths) == 0
        self.num_samples = num_synthetic_fallback if self.synthetic else len(self.file_paths)

    def __len__(self) -> int:
        return self.num_samples

    def _generate_synthetic_tissue(self) -> np.ndarray:
        x = np.linspace(0, 1, self.width)
        y = np.linspace(0, 1, self.height)
        xx, yy = np.meshgrid(x, y)
        freq = self.rng.uniform(2.0, 6.0)
        tissue = 0.5 + 0.3 * np.sin(freq * yy) * np.cos(1.0 * xx)
        texture = self.rng.normal(0.0, 0.03, (self.height, self.width))
        return np.clip(tissue + texture, 0.0, 1.0).astype(np.float32)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.synthetic:
            frame = self._generate_synthetic_tissue()
        else:
            p = self.file_paths[idx]
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                frame = self._generate_synthetic_tissue()
            else:
                if (img.shape[1], img.shape[0]) != (self.width, self.height):
                    img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
                frame = img.astype(np.float32) / 255.0

        tensor = torch.from_numpy(frame[np.newaxis, :, :])
        return tensor


def train_and_export_autoencoder(args):
    print("[+] Starting ConvAutoencoder Training Pipeline...")
    if not args.force_cpu and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif not args.force_cpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[*] Training Device: {device}")

    torch.manual_seed(args.seed)
    in_channels = 1
    h, w = args.height, args.width
    epochs = args.epochs
    batch_size = args.batch_size

    train_ds = HealthyTissueDataset(
        split_csv=args.train_csv,
        height=h,
        width=w,
        seed=args.seed,
    )
    val_ds = HealthyTissueDataset(
        split_csv=args.val_csv,
        height=h,
        width=w,
        seed=args.seed + 1,
    )
    print(f"[*] Dataset Mode: {'SYNTHETIC' if train_ds.synthetic else 'CLINICAL (Negative-Only)'} (Train: {len(train_ds)}, Val: {len(val_ds)})")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = ConvAutoencoder(in_channels=in_channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inputs = batch.to(device)
            # Apply slight input perturbation to learn denoising/normal baseline reconstruction
            noisy_inputs = torch.clamp(inputs + torch.randn_like(inputs) * 0.03, 0.0, 1.0)

            optimizer.zero_grad()
            recon = model(noisy_inputs)
            loss = criterion(recon, inputs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch)

        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_batch in val_loader:
                v_inputs = val_batch.to(device)
                v_recon = model(v_inputs)
                val_loss += criterion(v_recon, v_inputs).item() * len(val_batch)
        val_loss /= len(val_ds)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 2 == 0 or epoch == epochs:
            print(f"    Epoch [{epoch}/{epochs}] - Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[+] ConvAutoencoder Training Complete! Best Val MSE: {best_val_loss:.6f}")

    # Save checkpoint
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pt_path = output_dir / "conv_autoencoder_tissue_baseline.pt"
    torch.save(model.state_dict(), pt_path)
    print(f"[+] Saved PyTorch checkpoint: {pt_path}")

    # --- ONNX Export & Parity Verification ---
    model.eval()
    onnx_path = str(output_dir / "conv_autoencoder_tissue_baseline.onnx")

    print(f"[*] Exporting ConvAutoencoder to ONNX: {onnx_path}")
    export_pytorch_to_onnx(
        model=model,
        output_path=onnx_path,
        height=h,
        width=w,
        in_channels=in_channels,
        input_name="input_frame",
        output_name="reconstructed_baseline",
        dynamic_spatial=True,
        opset_version=17,
    )
    ok, max_diff, _ = verify_pytorch_onnx_parity(
        pytorch_model=model,
        onnx_path=onnx_path,
        height=h,
        width=w,
        in_channels=in_channels,
    )
    print(f"[+] Exported ONNX ({os.path.getsize(onnx_path) / 1024:.2f} KB) | Parity ok={ok} (max_diff={max_diff:.2e})")
    return onnx_path


def parse_args():
    parser = argparse.ArgumentParser(description="Train Stage 2 ConvAutoencoder on Negative Baselines")
    parser.add_argument("--train-csv", type=str, default="data/splits/negative_train.csv")
    parser.add_argument("--val-csv", type=str, default="data/splits/negative_val.csv")
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=str, default="exports")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train_and_export_autoencoder(parse_args())
