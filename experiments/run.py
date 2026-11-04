"""
experiments/run.py
==================
Pre-registered experiment runner executing training, temperature scaling, and evaluation
for front-end and downstream family configurations.

Appends results directly to `experiments/results.csv`:
    run_id, git_sha, front_end, family, seed, split, metric, value
"""

import os
import sys
import csv
import yaml
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.metrics import compute_ece, compute_macro_auroc
from models.classifier.mobilenet_v3 import MobileNetV3Classifier, fit_temperature
from models.classifier.scan_classifier import CalibratedScanClassifier, UnifiedScanClassifier
from models.classifier.train import CLASS_TO_INDEX, load_split
from src_python.onnx_model_io import OnnxImageAdapter
from src_python.pathology import crop_roi
from src_python.roi_logic import ROIDecisionLogic


def get_git_sha() -> str:
    """Retrieves current short git SHA or 'unknown'."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


class ScanDataset(Dataset):
    """Full-scan dataset with optional front-end preprocessing and horizontal flip."""

    def __init__(
        self,
        images: List[np.ndarray],
        labels: List[int],
        is_train: bool,
        front_end: str = "raw",
        scan_size: int = 256,
        seed: int = 42,
    ):
        self.images = images
        self.labels = labels
        self.is_train = is_train
        self.front_end = front_end.lower()
        self.scan_size = scan_size
        self.rng = np.random.default_rng(seed)

        self._adapter: Optional[OnnxImageAdapter] = None
        if self.front_end == "unet_synth":
            model_path = str(REPO_ROOT / "exports" / "unet_despeckler.onnx")
            if os.path.exists(model_path):
                self._adapter = OnnxImageAdapter(model_path)
        elif self.front_end == "unet_n2n":
            model_path = str(REPO_ROOT / "exports" / "unet_n2n.onnx")
            if os.path.exists(model_path):
                self._adapter = OnnxImageAdapter(model_path)

    def __len__(self) -> int:
        return len(self.images)

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        if self.front_end == "raw":
            return img
        elif self.front_end == "nlm":
            img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            denoised = cv2.fastNlMeansDenoising(img_u8, None, h=10, templateWindowSize=7, searchWindowSize=21)
            return denoised.astype(np.float32) / 255.0
        elif self.front_end == "bm3d":
            try:
                import bm3d
                denoised = bm3d.bm3d(img, sigma_psd=0.08, stage_arg=bm3d.BM3DStages.ALL_STAGES)
                return np.clip(denoised, 0.0, 1.0).astype(np.float32)
            except ImportError:
                return img
        elif self.front_end in ("unet_synth", "unet_n2n"):
            if self._adapter is not None:
                return self._adapter(img)
            return img
        return img

    def __getitem__(self, idx: int):
        img = self.images[idx]
        if self.is_train and self.rng.random() < 0.5:
            img = img[:, ::-1]
        proc = self._preprocess(img)
        tensor = torch.from_numpy(np.ascontiguousarray(proc[np.newaxis], dtype=np.float32))
        return tensor, self.labels[idx]


class MILBagDataset(Dataset):
    """
    Downstream Family (a): Multiple Instance Learning (MIL) dataset.
    Extracts candidate lesion ROI crops (64x64) proposed by ConvAutoencoder + ROIDecisionLogic.
    Each item is (crops_tensor [K, 1, 64, 64], scan_label).
    """

    def __init__(
        self,
        images: List[np.ndarray],
        labels: List[int],
        is_train: bool,
        front_end: str = "raw",
        ae_onnx_path: str = "exports/conv_autoencoder_tissue_baseline.onnx",
        max_crops_per_bag: int = 5,
        seed: int = 42,
    ):
        self.labels = labels
        self.is_train = is_train
        self.rng = np.random.default_rng(seed)

        ae_full = str(REPO_ROOT / ae_onnx_path if not os.path.isabs(ae_onnx_path) else ae_onnx_path)
        ae_adapter = OnnxImageAdapter(ae_full) if os.path.exists(ae_full) else None
        roi_logic = ROIDecisionLogic()

        fe_adapter = None
        if front_end == "unet_synth":
            p = str(REPO_ROOT / "exports" / "unet_despeckler.onnx")
            if os.path.exists(p):
                fe_adapter = OnnxImageAdapter(p)
        elif front_end == "unet_n2n":
            p = str(REPO_ROOT / "exports" / "unet_n2n.onnx")
            if os.path.exists(p):
                fe_adapter = OnnxImageAdapter(p)

        self.bags: List[torch.Tensor] = []
        for img in images:
            # 1. Front-end despeckling
            if front_end == "raw":
                despeckled = img
            elif front_end == "nlm":
                u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
                denoised = cv2.fastNlMeansDenoising(u8, None, h=10, templateWindowSize=7, searchWindowSize=21)
                despeckled = denoised.astype(np.float32) / 255.0
            elif front_end == "bm3d":
                try:
                    import bm3d
                    denoised = bm3d.bm3d(img, sigma_psd=0.08, stage_arg=bm3d.BM3DStages.ALL_STAGES)
                    despeckled = np.clip(denoised, 0.0, 1.0).astype(np.float32)
                except ImportError:
                    despeckled = img
            elif front_end in ("unet_synth", "unet_n2n") and fe_adapter is not None:
                despeckled = fe_adapter(img)
            else:
                despeckled = img

            # 2. Healthy reconstruction & anomaly heatmap
            if ae_adapter is not None:
                recon = ae_adapter(despeckled)
                mse_heatmap = (despeckled - recon) ** 2
            else:
                mse_heatmap = np.zeros_like(despeckled)

            findings, _ = roi_logic.detect_anomaly_rois(mse_heatmap)

            if findings:
                sorted_f = sorted(findings, key=lambda f: f.get("score", 0.0), reverse=True)[:max_crops_per_bag]
                crops = [crop_roi(despeckled, f["bbox"]) for f in sorted_f]
            else:
                peak_idx = np.unravel_index(np.argmax(mse_heatmap), mse_heatmap.shape)
                y, x = int(peak_idx[0]), int(peak_idx[1])
                x0 = max(0, min(x - 32, 256 - 64))
                y0 = max(0, min(y - 32, 256 - 64))
                crops = [crop_roi(despeckled, [x0, y0, 64, 64])]

            crop_tensors = []
            for c in crops:
                if is_train and self.rng.random() < 0.5:
                    c = c[:, ::-1]
                crop_tensors.append(torch.from_numpy(np.ascontiguousarray(c[np.newaxis], dtype=np.float32)))
            self.bags.append(torch.stack(crop_tensors, dim=0))

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.bags[idx], self.labels[idx]


def mil_collate_fn(batch):
    """Packages list of variable-size bag tensors and labels into a batch."""
    bags = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return bags, labels


def forward_batch(
    model: nn.Module,
    batch: Tuple,
    family: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes batch logits and target labels for either Family (a) or Family (b)."""
    if family == "a":
        bags, labels = batch
        labels = labels.to(device)
        flat_crops = torch.cat(bags, dim=0).to(device)
        flat_out = model(flat_crops)
        bag_sizes = [len(b) for b in bags]
        bag_logits = [out.max(dim=0).values for out in torch.split(flat_out, bag_sizes)]
        logits = torch.stack(bag_logits, dim=0)
        return logits, labels
    else:
        x, y = batch
        return model(x.to(device)), y.to(device)


def append_result_row(
    csv_path: str,
    run_id: str,
    git_sha: str,
    front_end: str,
    family: str,
    seed: int,
    split: str,
    metric: str,
    value: float,
):
    """Appends a single verified metric row to results.csv."""
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    fieldnames = ["run_id", "git_sha", "front_end", "family", "seed", "split", "metric", "value"]

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "run_id": run_id,
            "git_sha": git_sha,
            "front_end": front_end,
            "family": family,
            "seed": seed,
            "split": split,
            "metric": metric,
            "value": float(value),
        })


@torch.no_grad()
def collect_probabilities(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    family: str = "b",
    temperature: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collects all predicted probability distributions and ground-truth labels."""
    model.eval()
    all_probs = []
    all_labels = []
    for batch in loader:
        logits, labels = forward_batch(model, batch, family, device)
        probs = F.softmax(logits / temperature, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())
    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)


def run_experiment(config_path: str) -> Dict[str, float]:
    """Loads YAML config, trains and evaluates downstream model, and records results."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_id = cfg.get("run_id", Path(config_path).stem)
    front_end = cfg.get("front_end", "raw")
    family = cfg.get("family", "b").lower()
    seed = int(cfg.get("seed", 42))
    results_csv = cfg.get("results_csv", "experiments/results.csv")
    git_sha = get_git_sha()

    data_cfg = cfg.get("data", {})
    splits_dir = data_cfg.get("splits_dir", "data/splits")
    raw_root = Path(data_cfg.get("raw_root", "data/raw/kermany"))

    train_cfg = cfg.get("train", {})
    epochs = int(train_cfg.get("epochs", 15))
    batch_size = int(train_cfg.get("batch_size", 32))
    lr = float(train_cfg.get("lr", 1e-3))
    scan_size = int(train_cfg.get("scan_size", 256))

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[*] Starting run: {run_id} | Front-End: {front_end} | Family: {family} | Seed: {seed} | Device: {device}")

    # Load split scans
    train_imgs, train_lbls = load_split(os.path.join(splits_dir, "train.csv"), raw_root)
    val_imgs, val_lbls = load_split(os.path.join(splits_dir, "val.csv"), raw_root)
    test_csv = os.path.join(splits_dir, "test.csv")
    test_imgs, test_lbls = load_split(test_csv, raw_root) if os.path.exists(test_csv) else ([], [])

    if family == "a":
        ae_onnx = cfg.get("ae_onnx", "exports/conv_autoencoder_tissue_baseline.onnx")
        train_ds = MILBagDataset(train_imgs, train_lbls, is_train=True, front_end=front_end, ae_onnx_path=ae_onnx, seed=seed)
        val_ds = MILBagDataset(val_imgs, val_lbls, is_train=False, front_end=front_end, ae_onnx_path=ae_onnx, seed=seed)
        test_ds = MILBagDataset(test_imgs, test_lbls, is_train=False, front_end=front_end, ae_onnx_path=ae_onnx, seed=seed) if test_imgs else None

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=mil_collate_fn)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=mil_collate_fn)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=mil_collate_fn) if test_ds else None

        model = MobileNetV3Classifier(num_classes=4, in_channels=1).to(device)
    else:
        train_ds = ScanDataset(train_imgs, train_lbls, is_train=True, front_end=front_end, scan_size=scan_size, seed=seed)
        val_ds = ScanDataset(val_imgs, val_lbls, is_train=False, front_end=front_end, scan_size=scan_size, seed=seed)
        test_ds = ScanDataset(test_imgs, test_lbls, is_train=False, front_end=front_end, scan_size=scan_size, seed=seed) if test_imgs else None

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False) if test_ds else None

        model = UnifiedScanClassifier(num_classes=4, in_channels=1).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_macro_auroc = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            logits, targets = forward_batch(model, batch, family, device)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * targets.size(0)

        # Validation Macro AUROC
        val_probs, val_targets = collect_probabilities(model, val_loader, device, family=family, temperature=1.0)
        val_macro_auroc = compute_macro_auroc(val_targets, val_probs, num_classes=4)
        print(f"    Epoch [{epoch}/{epochs}] train_loss={running_loss / len(train_imgs):.4f} val_macro_auroc={val_macro_auroc:.4f}")

        if val_macro_auroc > best_val_macro_auroc:
            best_val_macro_auroc = val_macro_auroc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.to(device)

    # Temperature scaling on validation logits
    model.eval()
    val_raw_logits = []
    val_raw_targets = []
    with torch.no_grad():
        for batch in val_loader:
            logits, targets = forward_batch(model, batch, family, device)
            val_raw_logits.append(logits.cpu())
            val_raw_targets.append(targets.cpu())
    val_logits_t = torch.cat(val_raw_logits)
    val_targets_t = torch.cat(val_raw_targets).long()

    temp = fit_temperature(val_logits_t, val_targets_t)
    model.temperature.fill_(temp)
    print(f"[+] Fitted Temperature T = {temp:.4f}")

    # Evaluate on val
    val_probs_cal, val_targets_cal = collect_probabilities(model, val_loader, device, family=family, temperature=temp)
    val_final_auroc = compute_macro_auroc(val_targets_cal, val_probs_cal, num_classes=4)
    val_final_ece = compute_ece(val_targets_cal, val_probs_cal, n_bins=15)

    append_result_row(results_csv, run_id, git_sha, front_end, family, seed, "val", "macro_auroc", val_final_auroc)
    append_result_row(results_csv, run_id, git_sha, front_end, family, seed, "val", "ece", val_final_ece)

    metrics_summary = {
        "val_macro_auroc": val_final_auroc,
        "val_ece": val_final_ece,
    }

    # Evaluate on test if available
    if test_imgs:
        test_probs_cal, test_targets_cal = collect_probabilities(model, test_loader, device, family=family, temperature=temp)
        test_final_auroc = compute_macro_auroc(test_targets_cal, test_probs_cal, num_classes=4)
        test_final_ece = compute_ece(test_targets_cal, test_probs_cal, n_bins=15)

        append_result_row(results_csv, run_id, git_sha, front_end, family, seed, "test", "macro_auroc", test_final_auroc)
        append_result_row(results_csv, run_id, git_sha, front_end, family, seed, "test", "ece", test_final_ece)

        metrics_summary["test_macro_auroc"] = test_final_auroc
        metrics_summary["test_ece"] = test_final_ece
        print(f"[+] Test: macro_auroc={test_final_auroc:.4f} | ece={test_final_ece:.4f}")

    # Record model parameter efficiency
    num_params = sum(p.numel() for p in model.parameters())
    append_result_row(results_csv, run_id, git_sha, front_end, family, seed, "all", "params", float(num_params))
    metrics_summary["params"] = float(num_params)

    # Save checkpoint if requested
    ckpt_path = cfg.get("checkpoint_path")
    if ckpt_path:
        os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        print(f"[+] Saved checkpoint to {ckpt_path}")

    return metrics_summary


def main():
    parser = argparse.ArgumentParser(description="Run experiment configuration")
    parser.add_argument("config", type=str, help="Path to YAML experiment config")
    args = parser.parse_args()

    run_experiment(args.config)


if __name__ == "__main__":
    main()
