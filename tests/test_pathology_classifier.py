"""Tests for Stage 3: ROI crop extraction, pathology formatting, ScanSession classification,
and the MobileNetV3 classifier (model, temperature scaling, ONNX export + parity)."""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src_python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathology import PATHOLOGY_CLASSES, ROI_CROP_SIZE, crop_roi, describe_pathology


# ---------------------------------------------------------------------------
# Torch-free: crop extraction and formatting
# ---------------------------------------------------------------------------

def test_crop_roi_returns_fixed_size_square_crop_of_the_finding():
    frame = np.zeros((128, 128), np.float32)
    frame[30:70, 40:80] = 1.0  # 40x40 lesion
    crop = crop_roi(frame, [40, 30, 40, 40])
    assert crop.shape == (ROI_CROP_SIZE, ROI_CROP_SIZE) and crop.dtype == np.float32
    assert crop.min() > 0.99  # crop is entirely lesion


def test_crop_roi_non_square_box_is_expanded_not_stretched():
    frame = np.zeros((100, 100), np.float32)
    frame[40:60, 10:90] = 1.0  # wide 80x20 band
    crop = crop_roi(frame, [10, 40, 80, 20])
    # Square 80x80 window centred on the band: band covers the middle quarter of rows
    assert 0.15 < crop.mean() < 0.35


def test_crop_roi_clamps_to_frame_edges():
    frame = np.random.default_rng(0).random((64, 64)).astype(np.float32)
    crop = crop_roi(frame, [55, 55, 20, 20])  # runs off the bottom-right corner
    assert crop.shape == (ROI_CROP_SIZE, ROI_CROP_SIZE)
    crop_big = crop_roi(frame, [0, 0, 500, 500])  # larger than the frame
    assert crop_big.shape == (ROI_CROP_SIZE, ROI_CROP_SIZE)


def test_crop_roi_rejects_bad_inputs():
    with pytest.raises(ValueError):
        crop_roi(np.zeros((8, 8, 3), np.float32), [0, 0, 4, 4])
    with pytest.raises(ValueError):
        crop_roi(np.zeros((8, 8), np.float32), [0, 0, 0, 4])


def test_describe_pathology_picks_argmax_and_validates_length():
    d = describe_pathology([0.1, 0.2, 0.6, 0.1])
    assert d["label"] == "DRUSEN" and d["confidence"] == pytest.approx(0.6)
    assert set(d["probabilities"]) == set(PATHOLOGY_CLASSES)
    with pytest.raises(ValueError):
        describe_pathology([0.5, 0.5])


# ---------------------------------------------------------------------------
# ScanSession integration (needs compiled scan_core, no torch)
# ---------------------------------------------------------------------------

def test_scan_session_attaches_pathology_to_each_finding(tmp_path):
    pytest.importorskip("scan_core", reason="scan_core not built")
    from scan_session import ScanSession

    H = W = 128
    video = str(tmp_path / "scan.avi")
    writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (W, H), False)
    for _ in range(8):
        img = np.full((H, W), 128, np.uint8)
        img[30:70, 40:80] = 240
        writer.write(img)
    writer.release()

    seen_crops = []

    def fake_classifier(crop):
        seen_crops.append(crop)
        return np.array([0.05, 0.05, 0.85, 0.05], np.float32)

    logged = []
    with ScanSession(
        source=video,
        despeckler=lambda img: img,
        reconstructor=lambda img: np.full_like(img, 0.5, dtype=np.float32),
        on_finding=logged.append,
        classifier=fake_classifier,
    ) as session:
        res = None
        for _ in range(100):
            res = session.process_next()
            if res is not None:
                break
        assert res is not None and len(res.findings) == 1
        finding = res.findings[0]
        assert finding["pathology"]["label"] == "DRUSEN"
        assert logged[0]["pathology"]["label"] == "DRUSEN"  # classified BEFORE on_finding fires
        assert seen_crops[0].shape == (ROI_CROP_SIZE, ROI_CROP_SIZE)


# ---------------------------------------------------------------------------
# Torch: model, temperature scaling, ONNX
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch", reason="torch not installed")
pytest.importorskip("torchvision", reason="torchvision not installed")

from models.classifier.mobilenet_v3 import (  # noqa: E402
    CalibratedClassifier,
    MobileNetV3Classifier,
    fit_temperature,
)


def test_classifier_output_shape_and_probabilities_sum_to_one():
    model = MobileNetV3Classifier().eval()
    x = torch.rand(3, 1, 64, 64)
    assert model(x).shape == (3, len(PATHOLOGY_CLASSES))
    probs = model.predict_proba(x)
    assert torch.allclose(probs.sum(dim=1), torch.ones(3), atol=1e-5)


def test_classifier_gradient_flows_to_stem():
    model = MobileNetV3Classifier()
    loss = torch.nn.functional.cross_entropy(model(torch.rand(2, 1, 64, 64)), torch.tensor([0, 3]))
    loss.backward()
    stem = model.backbone.features[0][0]
    assert stem.weight.grad is not None and torch.isfinite(stem.weight.grad).all()


def test_higher_temperature_softens_probabilities():
    model = MobileNetV3Classifier().eval()
    x = torch.rand(4, 1, 64, 64)
    sharp = model.predict_proba(x).max(dim=1).values
    model.temperature.fill_(5.0)
    soft = model.predict_proba(x).max(dim=1).values
    assert (soft <= sharp + 1e-6).all()


def test_fit_temperature_reduces_nll_for_overconfident_logits():
    g = torch.Generator().manual_seed(0)
    labels = torch.randint(0, 4, (400,), generator=g)
    logits = torch.randn(400, 4, generator=g) * 0.5
    logits[torch.arange(400), labels] += 1.0
    logits = logits * 8.0  # overconfident: huge scale, many confident mistakes

    nll_before = torch.nn.functional.cross_entropy(logits, labels).item()
    t = fit_temperature(logits, labels)
    nll_after = torch.nn.functional.cross_entropy(logits / t, labels).item()
    assert t > 1.0
    assert nll_after < nll_before


def test_fit_temperature_validates_shapes():
    with pytest.raises(ValueError):
        fit_temperature(torch.zeros(5, 4), torch.zeros(3))


def test_onnx_export_parity_and_session_adapter(tmp_path):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    from onnx_model_io import OnnxCropClassifierAdapter, export_pytorch_to_onnx, verify_pytorch_onnx_parity

    model = MobileNetV3Classifier().eval()
    model.temperature.fill_(1.7)
    wrapped = CalibratedClassifier(model)
    onnx_path = str(tmp_path / "pathology.onnx")

    export_pytorch_to_onnx(
        wrapped, onnx_path, height=64, width=64,
        input_name="roi_crop", output_name="class_probabilities", dynamic_spatial=False,
    )
    ok, max_diff, _ = verify_pytorch_onnx_parity(wrapped, onnx_path, height=64, width=64)
    assert ok, f"ONNX parity failed (max_diff={max_diff})"

    adapter = OnnxCropClassifierAdapter(onnx_path)
    probs = adapter(np.random.default_rng(0).random((64, 64)).astype(np.float32))
    assert probs.shape == (len(PATHOLOGY_CLASSES),)
    assert probs.sum() == pytest.approx(1.0, abs=1e-5)
    # Wrong-sized crops are resized to the model's fixed input
    assert adapter(np.random.default_rng(1).random((90, 70)).astype(np.float32)).shape == (len(PATHOLOGY_CLASSES),)
