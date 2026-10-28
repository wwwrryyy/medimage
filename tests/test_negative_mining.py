"""Unit tests for Negative Mining & Healthy Baseline Dataset extraction."""

import os
import sys
import csv
from pathlib import Path
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_scripts.negative_mining import extract_negative_samples, verify_negative_isolation
from models.anomaly_detector.train_autoencoder import HealthyTissueDataset


def test_negative_mining_excludes_all_pathological_classes(tmp_path):
    # Create dummy split CSV containing normal and pathological entries
    split_csv = tmp_path / "dummy_split.csv"
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    (raw_root / "NORMAL").mkdir()
    (raw_root / "CNV").mkdir()

    # Create dummy image files
    norm_img = raw_root / "NORMAL" / "NORMAL-111-1.jpeg"
    norm_img.touch()
    cnv_img = raw_root / "CNV" / "CNV-222-1.jpeg"
    cnv_img.touch()

    with open(split_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "class", "patient_id"])
        writer.writeheader()
        writer.writerow({"image_path": str(norm_img), "class": "NORMAL", "patient_id": "111"})
        writer.writerow({"image_path": str(cnv_img), "class": "CNV", "patient_id": "222"})
        writer.writerow({"image_path": "dummy/path.jpeg", "class": "DME", "patient_id": "333"})

    records = extract_negative_samples(split_csv, raw_root)
    assert len(records) == 1
    assert records[0]["class"] == "NORMAL"
    assert records[0]["patient_id"] == "111"
    assert verify_negative_isolation(records) is True


def test_healthy_tissue_dataset_synthetic_fallback(tmp_path):
    # Empty or non-existent CSV should fall back to synthetic generation
    dataset = HealthyTissueDataset(split_csv=str(tmp_path / "nonexistent.csv"), num_synthetic_fallback=10)
    assert dataset.synthetic is True
    assert len(dataset) == 10
    sample = dataset[0]
    assert isinstance(sample, torch.Tensor)
    assert sample.shape == (1, 128, 128)
    assert sample.min() >= 0.0 and sample.max() <= 1.0


def test_healthy_tissue_dataset_loads_real_mined_split():
    neg_train_csv = Path("data/splits/negative_train.csv")
    if not neg_train_csv.exists():
        pytest.skip("negative_train.csv not generated")

    dataset = HealthyTissueDataset(split_csv=str(neg_train_csv), height=128, width=128)
    assert dataset.synthetic is False
    assert len(dataset) > 0
    sample = dataset[0]
    assert isinstance(sample, torch.Tensor)
    assert sample.shape == (1, 128, 128)
    assert sample.min() >= 0.0 and sample.max() <= 1.0
