"""
experiments/evaluate_localization.py
====================================
Evaluates lesion localization performance (AUPRO up to FPR 0.3, Dice) on fluid-annotated
clinical OCT volumes (Duke DME 2015, Chiu et al., BOE 2015) as pre-registered in EXPERIMENT_PLAN.md:

- Family (a) Localization: Pixel-wise MSE Anomaly Heatmap (despeckled - reconstruction)^2
- Family (b) Localization: Unified MobileNetV3 CAM upsampled to full scan resolution
- Front-ends: raw, nlm, bm3d, unet_synth, unet_n2n

Outputs:
    Appends rows to `experiments/results.csv`:
        run_id, git_sha, front_end, family, seed, split, metric, value
"""

import os
import sys
import csv
import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.metrics import compute_aupro, compute_dice_score
from experiments.run import append_result_row, get_git_sha
from src_python.onnx_model_io import OnnxImageAdapter
from models.classifier.scan_classifier import UnifiedScanClassifier


def evaluate_localization(
    manifest_csv: str = "data/splits/duke_dme_eval.csv",
    family: str = "a",
    front_end: str = "raw",
    seed: int = 42,
    results_csv: str = "experiments/results.csv",
    checkpoint: str = None,
    max_fpr: float = 0.30,
) -> dict:
    """Evaluates AUPRO@0.3 and optimal Dice on fluid-positive Duke DME volumes."""
    git_sha = get_git_sha()
    run_id = f"loc_{family}_{front_end}_seed{seed}"

    # Initialize front-end adapter
    fe_adapter = None
    if front_end == "unet_synth":
        p = str(REPO_ROOT / "exports" / "unet_despeckler.onnx")
        if os.path.exists(p):
            fe_adapter = OnnxImageAdapter(p)
    elif front_end == "unet_n2n":
        p = str(REPO_ROOT / "exports" / "unet_n2n.onnx")
        if os.path.exists(p):
            fe_adapter = OnnxImageAdapter(p)

    def preprocess(img: np.ndarray) -> np.ndarray:
        if front_end == "raw":
            return img
        elif front_end == "nlm":
            u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            denoised = cv2.fastNlMeansDenoising(u8, None, h=10, templateWindowSize=7, searchWindowSize=21)
            return denoised.astype(np.float32) / 255.0
        elif front_end == "bm3d":
            try:
                import bm3d
                denoised = bm3d.bm3d(img, sigma_psd=0.08, stage_arg=bm3d.BM3DStages.ALL_STAGES)
                return np.clip(denoised, 0.0, 1.0).astype(np.float32)
            except ImportError:
                return img
        elif front_end in ("unet_synth", "unet_n2n") and fe_adapter is not None:
            return fe_adapter(img)
        return img

    # Load models depending on family
    if family == "a":
        ae_path = str(REPO_ROOT / "exports" / "conv_autoencoder_tissue_baseline.onnx")
        if not os.path.exists(ae_path):
            raise FileNotFoundError(f"Autoencoder ONNX not found at {ae_path}")
        ae_adapter = OnnxImageAdapter(ae_path)
    else:
        # Family (b): Load UnifiedScanClassifier
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = UnifiedScanClassifier(num_classes=4, in_channels=1).to(device)
        if checkpoint and os.path.exists(checkpoint):
            model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.eval()

    gt_masks: List[np.ndarray] = []
    pred_heatmaps: List[np.ndarray] = []

    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["has_fluid"]) != 1:
                continue

            img = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            mask = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE) > 0

            despeckled = preprocess(img)

            if family == "a":
                recon = ae_adapter(despeckled)
                heatmap = (despeckled - recon) ** 2
            else:
                x_t = torch.from_numpy(despeckled[np.newaxis, np.newaxis]).to(device)
                # DME is class index 1
                cam_t = model.compute_cam(x_t, target_class=1)
                heatmap = cam_t[0].cpu().numpy()

            gt_masks.append(mask)
            pred_heatmaps.append(heatmap)

    if not gt_masks:
        print("[!] No fluid masks loaded for evaluation.")
        return {}

    print(f"[*] Loaded {len(gt_masks)} fluid slices from {manifest_csv}")

    # Compute AUPRO up to FPR 0.3
    aupro_val = compute_aupro(gt_masks, pred_heatmaps, max_fpr=max_fpr)

    # Compute best Dice score across sampled thresholds
    all_scores = np.concatenate([h.flatten() for h in pred_heatmaps])
    sample_th = np.linspace(all_scores.min(), all_scores.max(), 30)
    best_dice = 0.0
    for th in sample_th:
        dice_scores = [compute_dice_score(gt, h >= th) for gt, h in zip(gt_masks, pred_heatmaps)]
        m_dice = float(np.mean(dice_scores))
        if m_dice > best_dice:
            best_dice = m_dice

    print(f"[+] Localization Results ({run_id}): AUPRO@{max_fpr}={aupro_val:.4f} | Best Dice={best_dice:.4f}")

    # Append to results.csv
    append_result_row(results_csv, run_id, git_sha, front_end, family, seed, "duke_dme", "aupro", aupro_val)
    append_result_row(results_csv, run_id, git_sha, front_end, family, seed, "duke_dme", "dice", best_dice)

    return {
        "aupro": aupro_val,
        "dice": best_dice,
        "num_slices": len(gt_masks),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate lesion localization AUPRO/Dice")
    parser.add_argument("--manifest", type=str, default="data/splits/duke_dme_eval.csv")
    parser.add_argument("--family", type=str, default="a", choices=["a", "b"])
    parser.add_argument("--front-end", type=str, default="raw", choices=["raw", "nlm", "bm3d", "unet_synth", "unet_n2n"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--results-csv", type=str, default="experiments/results.csv")
    args = parser.parse_args()

    evaluate_localization(
        manifest_csv=args.manifest,
        family=args.family,
        front_end=args.front_end,
        seed=args.seed,
        results_csv=args.results_csv,
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
