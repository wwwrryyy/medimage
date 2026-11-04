"""
tests/test_conformal.py
=======================
Unit tests for split conformal prediction sets and clinical referral triage.
"""

import pytest
import numpy as np

from models.classifier.conformal import ConformalPredictor, ReferralCalibrator


def test_conformal_predictor_coverage():
    rng = np.random.default_rng(42)
    n_val = 500
    n_test = 500
    num_classes = 4

    # Generate synthetic softmax probabilities biased towards true class
    val_labels = rng.integers(0, num_classes, size=n_val)
    val_logits = rng.normal(0, 1, size=(n_val, num_classes))
    for i, y in enumerate(val_labels):
        val_logits[i, y] += 2.5
    val_probs = np.exp(val_logits) / np.sum(np.exp(val_logits), axis=1, keepdims=True)

    test_labels = rng.integers(0, num_classes, size=n_test)
    test_logits = rng.normal(0, 1, size=(n_test, num_classes))
    for i, y in enumerate(test_labels):
        test_logits[i, y] += 2.5
    test_probs = np.exp(test_logits) / np.sum(np.exp(test_logits), axis=1, keepdims=True)

    cp = ConformalPredictor(alpha=0.10)
    hat_q = cp.calibrate(val_probs, val_labels)
    assert 0.0 < hat_q < 1.0

    eval_res = cp.evaluate_coverage(test_probs, test_labels)
    # Empirical coverage should be approximately >= 90% (allowing small test sampling margin)
    assert eval_res["empirical_coverage"] >= 0.85
    assert 1.0 <= eval_res["avg_set_size"] <= num_classes


def test_conformal_predictor_uncalibrated_raises():
    cp = ConformalPredictor()
    with pytest.raises(RuntimeError):
        cp.predict_set(np.array([[0.25, 0.25, 0.25, 0.25]]))


def test_referral_calibrator():
    rng = np.random.default_rng(42)
    n = 400
    normal_idx = 3

    labels = rng.integers(0, 4, size=n)
    probs = np.zeros((n, 4))
    for i, y in enumerate(labels):
        if y == normal_idx:
            probs[i] = [0.05, 0.05, 0.05, 0.85]
        else:
            probs[i, y] = 0.80
            probs[i, normal_idx] = 0.10
            # Normalize
            probs[i] /= probs[i].sum()

    calibrator = ReferralCalibrator(normal_class_idx=normal_idx, target_sensitivity=0.95)
    th = calibrator.calibrate(probs[:200], labels[:200])
    assert 0.01 <= th <= 0.99

    # Test triage on clear pathology
    triage_path = calibrator.triage(np.array([0.90, 0.05, 0.03, 0.02]))
    assert triage_path["decision"] == "REFER"
    assert triage_path["needs_referral"] is True

    # Test triage on clear normal
    triage_norm = calibrator.triage(np.array([0.02, 0.02, 0.02, 0.94]))
    assert triage_norm["decision"] == "NO_REFERRAL"
    assert triage_norm["needs_referral"] is False

    # Test triage on uncertain / ambiguous case
    triage_ambig = calibrator.triage(np.array([0.20, 0.20, 0.20, 0.40]))
    assert triage_ambig["decision"] in ("REFER", "REFER_INDETERMINATE")
    assert triage_ambig["needs_referral"] is True
