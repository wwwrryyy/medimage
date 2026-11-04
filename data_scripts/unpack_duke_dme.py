"""
data_scripts/unpack_duke_dme.py
===============================
Extracts Duke DME 2015 (Chiu et al., BOE 2015) 3D OCT volumes and expert fluid segmentation masks:
    data/processed/duke_dme/Subject_{id:02d}/slice_{idx:03d}.png
    data/processed/duke_dme/Subject_{id:02d}/mask_{idx:03d}.png

Outputs:
    data/splits/duke_dme_eval.csv
"""

import os
import csv
import glob
import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat

TARGET_SIZE = 256


def unpack_subject(mat_file: Path, output_dir: Path) -> list:
    """Unpacks a single subject's .mat volume and fluid masks."""
    subj_name = mat_file.stem  # e.g. Subject_01
    subj_dir = output_dir / subj_name
    subj_dir.mkdir(parents=True, exist_ok=True)

    data = loadmat(str(mat_file))
    images = data["images"]  # [H, W, Slices]
    # Use manualFluid1 (Expert Grader 1) primary ground truth
    fluid_mask = data.get("manualFluid1")
    if fluid_mask is None:
        fluid_mask = data.get("manualFluid2")
    if fluid_mask is None:
        fluid_mask = np.zeros_like(images, dtype=np.float64)

    num_slices = images.shape[2]
    records = []

    for s in range(num_slices):
        img_slice = images[:, :, s]
        mask_slice = (fluid_mask[:, :, s] > 0).astype(np.uint8) * 255

        # Resize to standard 256x256 resolution
        img_resized = cv2.resize(img_slice, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
        mask_resized = cv2.resize(mask_slice, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_NEAREST)

        img_path = subj_dir / f"slice_{s:03d}.png"
        mask_path = subj_dir / f"mask_{s:03d}.png"

        cv2.imwrite(str(img_path), img_resized)
        cv2.imwrite(str(mask_path), mask_resized)

        has_fluid = int(np.any(mask_resized > 0))
        fluid_area = int(np.sum(mask_resized > 0))

        records.append({
            "subject_id": subj_name,
            "slice_idx": s,
            "image_path": str(img_path),
            "mask_path": str(mask_path),
            "has_fluid": has_fluid,
            "fluid_area": fluid_area,
        })

    return records


def unpack_all(raw_dir: Path, output_dir: Path, manifest_csv: Path):
    """Unpacks all subjects and saves evaluation manifest CSV."""
    mat_files = sorted(list(raw_dir.glob("Subject_*.mat")))
    if not mat_files:
        raise RuntimeError(f"No Subject_*.mat files found in {raw_dir}")

    print(f"[*] Unpacking {len(mat_files)} Duke DME 2015 subjects into {output_dir}...")
    all_records = []

    for mf in mat_files:
        records = unpack_subject(mf, output_dir)
        fluid_cnt = sum(r["has_fluid"] for r in records)
        print(f"    - {mf.stem}: {len(records)} slices, {fluid_cnt} with fluid lesions")
        all_records.extend(records)

    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["subject_id", "slice_idx", "image_path", "mask_path", "has_fluid", "fluid_area"]
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    total_fluid = sum(r["has_fluid"] for r in all_records)
    print(f"\n[+] Extracted {len(all_records)} total slices ({total_fluid} with fluid)")
    print(f"[+] Manifest saved to: {manifest_csv}")


def main():
    parser = argparse.ArgumentParser(description="Unpack Duke DME 2015 fluid segmentation dataset")
    parser.add_argument("--raw-dir", type=str, default="data/raw/duke_dme_2015/2015_BOE_Chiu")
    parser.add_argument("--output-dir", type=str, default="data/processed/duke_dme")
    parser.add_argument("--manifest", type=str, default="data/splits/duke_dme_eval.csv")
    args = parser.parse_args()

    unpack_all(Path(args.raw_dir), Path(args.output_dir), Path(args.manifest))


if __name__ == "__main__":
    main()
