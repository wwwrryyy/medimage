"""Unit tests for ROIDecisionLogic (single threshold ownership, NMS, calibrated confidence)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src_python"))
from roi_logic import ROIDecisionLogic


def test_threshold_strictly_gates_findings():
    heatmap = np.zeros((100, 100), dtype=np.float32)
    heatmap[20:50, 20:50] = 0.07  # 30x30 = 900 px

    low_logic = ROIDecisionLogic(mse_threshold=0.05)
    findings_low, _ = low_logic.detect_anomaly_rois(heatmap)
    assert len(findings_low) == 1
    assert findings_low[0]["type"] == "MEDIUM ANOMALY REGION"

    high_logic = ROIDecisionLogic(mse_threshold=0.08)
    findings_high, _ = high_logic.detect_anomaly_rois(heatmap)
    assert findings_high == []


def test_calibrated_confidence_is_monotonic_and_anchored_at_half():
    logic = ROIDecisionLogic(mse_threshold=0.06, temperature=0.5)
    assert logic.calibrate_confidence(0.06) == pytest.approx(0.5, abs=1e-6)

    c_low = logic.calibrate_confidence(0.07)
    c_mid = logic.calibrate_confidence(0.12)
    c_high = logic.calibrate_confidence(0.30)
    assert 0.5 < c_low < c_mid < c_high < 1.0

    # Higher temperature softens confidence toward 0.5
    softer = ROIDecisionLogic(mse_threshold=0.06, temperature=2.0)
    assert softer.calibrate_confidence(0.12) < logic.calibrate_confidence(0.12)


def test_nms_suppresses_overlapping_finding_boxes():
    logic = ROIDecisionLogic(mse_threshold=0.06, nms_iou_threshold=0.3)
    boxes = [[10, 10, 40, 40], [12, 12, 40, 40], [70, 70, 20, 20]]
    scores = [0.20, 0.35, 0.15]

    kept_boxes, kept_scores = logic.apply_nms(boxes, scores)
    assert kept_boxes == [[12, 12, 40, 40], [70, 70, 20, 20]]
    assert kept_scores == [0.35, 0.15]


def test_invalid_threshold_or_temperature_rejected():
    with pytest.raises(ValueError):
        ROIDecisionLogic(mse_threshold=0.0)
    with pytest.raises(ValueError):
        ROIDecisionLogic(temperature=-1.0)
