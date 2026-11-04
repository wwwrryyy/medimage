"""
tests/test_experiment_runner.py
===============================
Tests verifying experiment runner configuration parsing, ScanDataset preprocessing,
results row persistence, and end-to-end execution.
"""

import os
import csv
import yaml
import numpy as np
import pytest
import torch

from experiments.run import (
    MILBagDataset,
    ScanDataset,
    append_result_row,
    forward_batch,
    get_git_sha,
    mil_collate_fn,
    run_experiment,
)


def test_get_git_sha():
    sha = get_git_sha()
    assert isinstance(sha, str)
    assert len(sha) >= 4 or sha == "unknown"


def test_scan_dataset_preprocessing_and_shapes():
    # 2 dummy scans 64x64
    images = [np.ones((64, 64), dtype=np.float32), np.zeros((64, 64), dtype=np.float32)]
    labels = [0, 1]

    # raw
    ds_raw = ScanDataset(images, labels, is_train=False, front_end="raw")
    t0, l0 = ds_raw[0]
    assert t0.shape == (1, 64, 64)
    assert l0 == 0

    # nlm
    ds_nlm = ScanDataset(images, labels, is_train=False, front_end="nlm")
    t1, l1 = ds_nlm[1]
    assert t1.shape == (1, 64, 64)
    assert l1 == 1


def test_append_result_row(tmp_path):
    csv_path = str(tmp_path / "test_results.csv")
    append_result_row(
        csv_path=csv_path,
        run_id="run_test",
        git_sha="abc1234",
        front_end="raw",
        family="b",
        seed=42,
        split="test",
        metric="macro_auroc",
        value=0.885,
    )

    assert os.path.exists(csv_path)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 1
        assert reader[0]["run_id"] == "run_test"
        assert reader[0]["metric"] == "macro_auroc"
        assert float(reader[0]["value"]) == 0.885


def test_run_experiment_on_synthetic_split(tmp_path, monkeypatch):
    splits_dir = tmp_path / "splits"
    raw_root = tmp_path / "raw"
    splits_dir.mkdir(parents=True)
    raw_root.mkdir(parents=True)

    # Create dummy images on disk
    import cv2
    img_paths = []
    for cls in ["CNV", "DME", "DRUSEN", "NORMAL"]:
        cls_dir = raw_root / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(2):
            p = cls_dir / f"{cls}-P{i}-1.jpeg"
            dummy_img = (np.random.rand(64, 64) * 255).astype(np.uint8)
            cv2.imwrite(str(p), dummy_img)
            img_paths.append((str(p), cls, f"P{i}"))

    # Write small train, val, test splits
    for split_name in ["train", "val", "test"]:
        with open(splits_dir / f"{split_name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image_path", "class", "patient_id"])
            for p, cls, pid in img_paths:
                w.writerow([p, cls, pid])

    # Write config
    cfg_path = tmp_path / "config.yaml"
    res_csv = tmp_path / "results.csv"
    cfg = {
        "run_id": "test_synthetic",
        "front_end": "raw",
        "family": "b",
        "seed": 42,
        "results_csv": str(res_csv),
        "data": {
            "splits_dir": str(splits_dir),
            "raw_root": str(raw_root),
        },
        "train": {
            "epochs": 1,
            "batch_size": 4,
            "lr": 0.001,
            "scan_size": 64,
        },
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

    summary = run_experiment(str(cfg_path))
    assert "val_macro_auroc" in summary
    assert "val_ece" in summary
    assert "test_macro_auroc" in summary
    assert "params" in summary
    assert os.path.exists(res_csv)


def test_mil_bag_dataset_and_forward_batch():
    from models.classifier.mobilenet_v3 import MobileNetV3Classifier

    images = [np.random.rand(256, 256).astype(np.float32) for _ in range(3)]
    labels = [0, 1, 3]

    ds = MILBagDataset(images, labels, is_train=True, front_end="raw")
    assert len(ds) == 3
    bag0, l0 = ds[0]
    assert bag0.ndim == 4
    assert bag0.shape[1:] == (1, 64, 64)
    assert l0 == 0

    batch = [ds[0], ds[1], ds[2]]
    collated = mil_collate_fn(batch)
    assert len(collated[0]) == 3
    assert collated[1].shape == (3,)

    model = MobileNetV3Classifier(num_classes=4, in_channels=1)
    device = torch.device("cpu")
    logits, targets = forward_batch(model, collated, family="a", device=device)
    assert logits.shape == (3, 4)
    assert targets.shape == (3,)


def test_run_experiment_family_a_synthetic(tmp_path):
    splits_dir = tmp_path / "splits"
    raw_root = tmp_path / "raw"
    splits_dir.mkdir(parents=True)
    raw_root.mkdir(parents=True)

    import cv2
    img_paths = []
    for cls in ["CNV", "DME", "DRUSEN", "NORMAL"]:
        cls_dir = raw_root / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(2):
            p = cls_dir / f"{cls}-P{i}-1.jpeg"
            dummy_img = (np.random.rand(64, 64) * 255).astype(np.uint8)
            cv2.imwrite(str(p), dummy_img)
            img_paths.append((str(p), cls, f"P{i}"))

    for split_name in ["train", "val", "test"]:
        with open(splits_dir / f"{split_name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image_path", "class", "patient_id"])
            for p, cls, pid in img_paths:
                w.writerow([p, cls, pid])

    cfg_path = tmp_path / "config_fam_a.yaml"
    res_csv = tmp_path / "results_fam_a.csv"
    cfg = {
        "run_id": "test_synthetic_family_a",
        "front_end": "raw",
        "family": "a",
        "seed": 42,
        "results_csv": str(res_csv),
        "data": {
            "splits_dir": str(splits_dir),
            "raw_root": str(raw_root),
        },
        "train": {
            "epochs": 1,
            "batch_size": 4,
            "lr": 0.001,
            "scan_size": 64,
        },
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

    summary = run_experiment(str(cfg_path))
    assert "val_macro_auroc" in summary
    assert "val_ece" in summary
    assert "test_macro_auroc" in summary
    assert "params" in summary
    assert os.path.exists(res_csv)
    with open(res_csv) as f:
        content = f.read()
        assert "family_a" in content or ",a," in content

