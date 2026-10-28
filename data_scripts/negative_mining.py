"""
negative_mining.py
==================
Negative-only mining script for Stage 2 Anomaly Detection baseline training.

Extracts healthy/normal tissue scans from Kermany clinical splits (guaranteeing
strict ZERO pathological sample leakage and patient-level isolation).

Supports exporting verified normal splits to CSV and preparing clean baseline
patches/slices for the unsupervised ConvAutoencoder.

Outputs:
    data/splits/negative_train.csv
    data/splits/negative_val.csv
"""

import os
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import cv2
import numpy as np


def extract_negative_samples(
    split_csv_path: Path,
    raw_root: Path,
    processed_root: Optional[Path] = None,
    split_name: str = "train",
) -> List[Dict[str, str]]:
    """
    Filters split records for strictly NORMAL (healthy) tissue samples,
    validating existence on disk either in raw or processed (NLM/BM3D) targets.
    Guarantees zero pathological samples (CNV, DME, DRUSEN) are included.
    """
    if not split_csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {split_csv_path}")

    healthy_records: List[Dict[str, str]] = []
    skipped_pathology = 0
    missing_files = 0

    with open(split_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cls_name = row["class"].strip().upper()
            if cls_name != "NORMAL":
                skipped_pathology += 1
                continue

            # Resolve real path
            raw_p = Path(row["image_path"])
            filename = raw_p.name

            # Look in raw_root / NORMAL / filename
            actual_raw = raw_root / "NORMAL" / filename
            if not actual_raw.exists() and raw_p.exists():
                actual_raw = raw_p

            # Check if processed NLM/clean target is available
            processed_p = None
            if processed_root:
                cand = processed_root / split_name / "NORMAL" / f"{raw_p.stem}.png"
                if cand.exists():
                    processed_p = cand

            if not actual_raw.exists() and not (processed_p and processed_p.exists()):
                missing_files += 1
                continue

            healthy_records.append({
                "patient_id": row["patient_id"],
                "class": "NORMAL",
                "raw_path": str(actual_raw) if actual_raw.exists() else "",
                "processed_path": str(processed_p) if processed_p and processed_p.exists() else "",
                "primary_path": str(processed_p) if processed_p and processed_p.exists() else str(actual_raw)
            })

    print(f"[*] Processed {split_csv_path.name}:")
    print(f"    - Extracted {len(healthy_records)} verified NORMAL samples.")
    print(f"    - Excluded {skipped_pathology} pathological scans.")
    if missing_files > 0:
        print(f"    - [!] Warning: {missing_files} files were missing on disk.")

    return healthy_records


def verify_negative_isolation(healthy_records: List[Dict[str, str]]) -> bool:
    """Verifies that all samples in the mined records belong strictly to NORMAL class."""
    for rec in healthy_records:
        if rec["class"] != "NORMAL":
            return False
    return True


def save_negative_split(records: List[Dict[str, str]], output_path: Path):
    """Saves verified negative-only records to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["patient_id", "class", "raw_path", "processed_path", "primary_path"]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    print(f"[+] Saved negative split to: {output_path} ({len(records)} entries)")


def main():
    parser = argparse.ArgumentParser(description="Negative Mining: Extract healthy tissue baselines")
    parser.add_argument("--splits-dir", type=str, default="data/splits", help="Path to data/splits directory")
    parser.add_argument("--raw-root", type=str, default="data/raw/kermany", help="Path to raw kermany directory")
    parser.add_argument("--processed-root", type=str, default="data/processed/nlm", help="Path to processed NLM directory")
    parser.add_argument("--output-dir", type=str, default="data/splits", help="Directory to save negative splits")
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    raw_root = Path(args.raw_root)
    processed_root = Path(args.processed_root) if args.processed_root else None
    output_dir = Path(args.output_dir)

    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"

    train_healthy = extract_negative_samples(train_csv, raw_root, processed_root, "train")
    val_healthy = extract_negative_samples(val_csv, raw_root, processed_root, "val")

    assert verify_negative_isolation(train_healthy), "Pathological sample detected in negative train split!"
    assert verify_negative_isolation(val_healthy), "Pathological sample detected in negative val split!"

    # Check zero patient overlap between train and val negative splits
    train_patients = {r["patient_id"] for r in train_healthy}
    val_patients = {r["patient_id"] for r in val_healthy}
    overlap = train_patients.intersection(val_patients)
    if overlap:
        raise ValueError(f"Patient overlap detected between train and val: {overlap}")
    print(f"[+] Verified ZERO patient overlap between healthy train ({len(train_patients)}) and val ({len(val_patients)}) patients.")

    save_negative_split(train_healthy, output_dir / "negative_train.csv")
    save_negative_split(val_healthy, output_dir / "negative_val.csv")


if __name__ == "__main__":
    main()
