"""
tests/test_experiment_metrics.py
================================
Unit tests for pre-registered experimental evaluation metrics:
- Macro AUROC
- Expected Calibration Error (ECE)
- Area Under Per-Region Overlap (AUPRO)
- Dice score
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from experiments.metrics import (
    compute_binary_auroc,
    compute_macro_auroc,
    compute_ece,
    compute_dice_score,
    compute_aupro,
)


def test_binary_auroc_perfect_and_inverse():
    y_true = np.array([0, 0, 1, 1])
    y_score_perfect = np.array([0.1, 0.2, 0.8, 0.9])
    assert compute_binary_auroc(y_true, y_score_perfect) == pytest.approx(1.0)

    y_score_inverse = np.array([0.9, 0.8, 0.2, 0.1])
    assert compute_binary_auroc(y_true, y_score_inverse) == pytest.approx(0.0)

    y_score_tied = np.array([0.5, 0.5, 0.5, 0.5])
    assert compute_binary_auroc(y_true, y_score_tied) == pytest.approx(0.5)


def test_macro_auroc_multiclass():
    y_true = np.array([0, 1, 2, 3])
    # Identity matrix probabilities -> perfect prediction
    y_prob = np.eye(4)
    macro_auc = compute_macro_auroc(y_true, y_prob, num_classes=4)
    assert macro_auc == pytest.approx(1.0)


def test_ece_perfect_calibration():
    # Perfectly calibrated predictions
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([
        [0.0, 1.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ])
    ece = compute_ece(y_true, y_prob, n_bins=10)
    assert ece == pytest.approx(0.0, abs=1e-5)


def test_dice_score():
    mask_a = np.array([[1, 1], [0, 0]])
    mask_b = np.array([[1, 1], [0, 0]])
    assert compute_dice_score(mask_a, mask_b) == pytest.approx(1.0)

    mask_c = np.array([[0, 0], [1, 1]])
    assert compute_dice_score(mask_a, mask_c) == pytest.approx(0.0, abs=1e-4)


def test_aupro_synthetic_regions():
    gt = np.zeros((32, 32), dtype=np.uint8)
    gt[10:20, 10:20] = 1  # 10x10 lesion region

    # Heatmap with high values on lesion
    pred_good = np.zeros((32, 32), dtype=np.float32)
    pred_good[10:20, 10:20] = 0.9

    aupro = compute_aupro([gt], [pred_good], max_fpr=0.30)
    assert aupro > 0.80
