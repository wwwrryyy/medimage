"""
tests/test_scan_classifier.py
=============================
Tests for Downstream Family (b) Unified MobileNetV3 scan classifier and CAM localization.
"""

import pytest
import torch
from models.classifier.scan_classifier import CalibratedScanClassifier, UnifiedScanClassifier
from models.classifier.mobilenet_v3 import fit_temperature


def test_unified_scan_classifier_forward_shape():
    model = UnifiedScanClassifier(num_classes=4, in_channels=1)
    model.eval()
    x = torch.randn(2, 1, 256, 256)
    logits = model(x)
    assert logits.shape == (2, 4)
    probs = model.predict_proba(x)
    assert probs.shape == (2, 4)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)


def test_unified_scan_classifier_cam_generation():
    model = UnifiedScanClassifier(num_classes=4, in_channels=1)
    x = torch.randn(3, 1, 128, 128)
    
    # 1. Automatic target selection (predicted class)
    cam_auto = model.compute_cam(x)
    assert cam_auto.shape == (3, 128, 128)
    assert cam_auto.min() >= 0.0
    assert cam_auto.max() <= 1.0 + 1e-5

    # 2. Specific integer class target
    cam_class1 = model.compute_cam(x, target_class=1)
    assert cam_class1.shape == (3, 128, 128)

    # 3. Per-sample tensor class target
    targets = torch.tensor([0, 2, 3])
    cam_tensor = model.compute_cam(x, target_class=targets)
    assert cam_tensor.shape == (3, 128, 128)


def test_unified_scan_classifier_temperature_scaling():
    model = UnifiedScanClassifier(num_classes=4, in_channels=1)
    logits = torch.randn(10, 4)
    labels = torch.randint(0, 4, (10,))
    t = fit_temperature(logits, labels)
    assert t > 0.0
    model.temperature.fill_(t)
    
    x = torch.randn(2, 1, 64, 64)
    calibrated = CalibratedScanClassifier(model)
    probs = calibrated(x)
    assert probs.shape == (2, 4)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)
