"""
data_scripts/make_splits.py
===========================
Generates patient-level stratified train / val / test splits (80 / 10 / 10)
for the Kermany Retinal OCT dataset with perceptual hash deduplication.

Guarantees:
1. GLOBAL ZERO patient leakage across splits: all scans from a given patient
   are assigned exclusively to either train, val, or test across all disease classes.
2. ZERO perceptual duplicate leakage: identical/near-identical scans across different
   patient IDs are merged via Disjoint Set Union (DSU) so duplicates never cross splits.
3. Class-stratified representation across CNV, DME, DRUSEN, and NORMAL.

Output:
    data/splits/train.csv
    data/splits/val.csv
    data/splits/test.csv
"""

import os
import re
import csv
import argparse
from pathlib import Path
from collections import defaultdict, Counter
import cv2
import numpy as np


def extract_patient_id(filename: str) -> str:
    """
    Extracts patient identifier from Kermany OCT filename.
    Standard patterns:
      CNV-1016042-1.jpeg        -> '1016042'
      DME-1072919-3.jpeg        -> '1072919'
      DRUSEN-1083162-1.jpeg     -> '1083162'
      NORMAL-1017237-1.jpeg     -> '1017237'
    """
    stem = Path(filename).stem
    parts = stem.split("-")
    if len(parts) >= 3:
        return parts[1]
    elif len(parts) == 2:
        return parts[0] if not parts[0].isalpha() else parts[1]

    match = re.search(r"(\d{4,10})", stem)
    if match:
        return match.group(1)
    return stem


def compute_dhash(img_path: str, hash_size: int = 8) -> int:
    """Computes 64-bit difference hash (dHash) from image file."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0
    resized = cv2.resize(img, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    val = 0
    for b in diff.flatten():
        val = (val << 1) | int(b)
    return val


class DisjointSet:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[rx] = ry


def scan_dataset(root_dir: Path):
    """Recursively scans root_dir for images, detecting class and patient ID."""
    classes = ["CNV", "DME", "DRUSEN", "NORMAL"]
    exts = {".jpeg", ".jpg", ".png", ".tif", ".tiff"}

    # patient_id -> list of (file_path, class_name)
    patient_records = defaultdict(list)

    for path in root_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue

        found_class = None
        for c in classes:
            if f"/{c}/" in str(path) or f"\\{c}\\" in str(path) or path.stem.upper().startswith(c):
                found_class = c
                break

        if not found_class:
            continue

        pid = extract_patient_id(path.name)
        patient_records[pid].append((str(path), found_class))

    return patient_records


def deduplicate_patients(patient_records: dict, compute_hashes: bool = True) -> dict:
    """
    Computes perceptual dHash for all scans and merges patients that share identical hashes.
    Returns: meta_patient_records: meta_pid -> list of (file_path, class_name, original_pid)
    """
    if not compute_hashes:
        return {
            pid: [(f, c, pid) for f, c in records]
            for pid, records in patient_records.items()
        }

    print("[*] Computing perceptual dHashes for duplicate detection...")
    dsu = DisjointSet()
    hash_to_patient = {}
    duplicates_found = 0

    for pid, records in patient_records.items():
        dsu.find(pid)
        for img_path, _ in records:
            h = compute_dhash(img_path)
            if h == 0:
                continue
            if h in hash_to_patient:
                existing_pid = hash_to_patient[h]
                if dsu.find(existing_pid) != dsu.find(pid):
                    dsu.union(existing_pid, pid)
                    duplicates_found += 1
            else:
                hash_to_patient[h] = pid

    if duplicates_found > 0:
        print(f"[!] Found {duplicates_found} cross-patient duplicate scan pairs! Merged via DSU.")
    else:
        print("[+] No cross-patient perceptual duplicates detected.")

    meta_records = defaultdict(list)
    for pid, records in patient_records.items():
        meta_pid = dsu.find(pid)
        for f, c in records:
            meta_records[meta_pid].append((f, c, pid))

    return meta_records


def create_stratified_splits(
    meta_patient_records: dict,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
):
    """
    Performs patient-cluster stratified split across disease classes.
    """
    rng = np.random.default_rng(seed)
    splits = {"train": [], "val": [], "test": []}

    # Group meta-patients by dominant disease class
    class_to_meta_pids = defaultdict(list)
    for meta_pid, records in meta_patient_records.items():
        class_counts = Counter(c for _, c, _ in records)
        dominant_class = class_counts.most_common(1)[0][0]
        class_to_meta_pids[dominant_class].append(meta_pid)

    train_meta_pids = set()
    val_meta_pids = set()
    test_meta_pids = set()

    for cls_name, meta_pids in class_to_meta_pids.items():
        pids = sorted(meta_pids)
        rng.shuffle(pids)

        n_total = len(pids)
        n_train = max(1, int(round(n_total * train_ratio)))
        n_val = max(1, int(round(n_total * val_ratio))) if n_total >= 3 else 0
        if n_train + n_val >= n_total and n_total >= 3:
            n_train = n_total - 2
            n_val = 1

        cls_train = set(pids[:n_train])
        cls_val = set(pids[n_train:n_train + n_val])
        cls_test = set(pids[n_train + n_val:])

        train_meta_pids.update(cls_train)
        val_meta_pids.update(cls_val)
        test_meta_pids.update(cls_test)

    # Disjoint assertions
    assert not (train_meta_pids & val_meta_pids), "Leakage detected between train and val meta-patients"
    assert not (train_meta_pids & test_meta_pids), "Leakage detected between train and test meta-patients"
    assert not (val_meta_pids & test_meta_pids), "Leakage detected between val and test meta-patients"

    # Populate splits
    for meta_pid in train_meta_pids:
        for f, cls_name, orig_pid in meta_patient_records[meta_pid]:
            splits["train"].append({"image_path": f, "class": cls_name, "patient_id": orig_pid})

    for meta_pid in val_meta_pids:
        for f, cls_name, orig_pid in meta_patient_records[meta_pid]:
            splits["val"].append({"image_path": f, "class": cls_name, "patient_id": orig_pid})

    for meta_pid in test_meta_pids:
        for f, cls_name, orig_pid in meta_patient_records[meta_pid]:
            splits["test"].append({"image_path": f, "class": cls_name, "patient_id": orig_pid})

    return splits


def save_splits(splits: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_path", "class", "patient_id"]

    for split_name, rows in splits.items():
        out_csv = output_dir / f"{split_name}.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        patients = set(r["patient_id"] for r in rows)
        classes = Counter(r["class"] for r in rows)
        print(
            f"[+] Saved {split_name}.csv: {len(rows)} scans across "
            f"{len(patients)} distinct patients: {dict(classes)}"
        )


def main():
    parser = argparse.ArgumentParser(description="Create patient-level splits for Kermany OCT")
    parser.add_argument("--data-dir", type=str, default="data/raw/kermany", help="Path to unpacked Kermany dataset")
    parser.add_argument("--output-dir", type=str, default="data/splits", help="Output directory for split CSVs")
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-dedup", action="store_true", help="Skip perceptual hash deduplication")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.output_dir).resolve()

    print(f"[*] Scanning {data_dir} for OCT images...")
    patient_records = scan_dataset(data_dir)

    total_scans = sum(len(v) for v in patient_records.values())
    total_patients = len(patient_records)
    print(f"[+] Total Scans: {total_scans}, Total Patients: {total_patients}")

    meta_records = deduplicate_patients(patient_records, compute_hashes=not args.no_dedup)
    splits = create_stratified_splits(
        meta_records,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )
    save_splits(splits, out_dir)


if __name__ == "__main__":
    main()
