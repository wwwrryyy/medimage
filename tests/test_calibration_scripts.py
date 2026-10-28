"""Unit tests for clinical confidence readout and NMS clustering calibration."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.calibrate_confidence_nms import (
    calibrate_parameters,
    fit_logistic_temperature,
)


def test_fit_logistic_temperature():
    # True positives with high score, false positives with lower score
    scores = np.array([0.08, 0.12, 0.15, 0.20, 0.05, 0.06], dtype=np.float32)
    labels = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    t = fit_logistic_temperature(scores, labels, mse_threshold=0.06, init_temp=1.0)
    assert t > 0.0
    assert np.isfinite(t)


def test_calibrate_parameters_synthetic_data():
    normal_hms = [np.full((64, 64), 0.02, dtype=np.float32) for _ in range(5)]
    path_hms = []
    for _ in range(5):
        hm = np.full((64, 64), 0.02, dtype=np.float32)
        hm[20:40, 20:40] = 0.10  # 400px lesion
        path_hms.append(hm)

    calib = calibrate_parameters(normal_hms, path_hms, target_sensitivity=0.80)
    assert "mse_threshold" in calib
    assert "temperature" in calib
    assert "nms_iou_threshold" in calib
    assert 0.02 <= calib["mse_threshold"] <= 0.10
    assert calib["temperature"] > 0.0
