"""
tests/test_localization_eval.py
===============================
Tests for Duke DME fluid localization evaluation (AUPRO and Dice).
"""

import os
import csv
import numpy as np
import pytest
import cv2

from experiments.evaluate_localization import evaluate_localization


def test_evaluate_localization_synthetic(tmp_path):
    manifest_csv = tmp_path / "eval.csv"
    results_csv = tmp_path / "results.csv"
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Create 2 mock image/mask pairs
    img1 = (np.random.rand(256, 256) * 255).astype(np.uint8)
    mask1 = np.zeros((256, 256), dtype=np.uint8)
    mask1[100:150, 100:150] = 255  # 50x50 fluid lesion

    img1_p = data_dir / "slice_001.png"
    mask1_p = data_dir / "mask_001.png"
    cv2.imwrite(str(img1_p), img1)
    cv2.imwrite(str(mask1_p), mask1)

    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subject_id", "slice_idx", "image_path", "mask_path", "has_fluid", "fluid_area"])
        w.writerow(["Subject_01", 1, str(img1_p), str(mask1_p), 1, 2500])

    res = evaluate_localization(
        manifest_csv=str(manifest_csv),
        family="a",
        front_end="raw",
        seed=42,
        results_csv=str(results_csv),
    )

    assert "aupro" in res
    assert "dice" in res
    assert 0.0 <= res["aupro"] <= 1.0
    assert 0.0 <= res["dice"] <= 1.0
    assert os.path.exists(results_csv)
