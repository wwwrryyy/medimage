"""Tests for the compiled scan_core C++ module (ScanEngine / zero-copy bridge)."""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src_python"))
scan_core = pytest.importorskip("scan_core", reason="scan_core not built (see src_cpp/CMakeLists.txt)")

H, W = 64, 64


@pytest.fixture
def engine(tmp_path):
    video = str(tmp_path / "scan.avi")
    writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (W, H), False)
    for i in range(10):
        writer.write(np.full((H, W), 20 * i, np.uint8))
    writer.release()

    eng = scan_core.ScanEngine()
    eng.start_stream(video)
    yield eng
    eng.stop_stream()


def next_frame(engine):
    for _ in range(100):
        ok, frame_id, raw = engine.next_frame()
        if ok:
            return frame_id, raw
    pytest.fail("no frame received from stream")


def test_heatmap_is_between_despeckled_and_recon(engine):
    frame_id, raw = next_frame(engine)
    assert raw.shape[:2] == (H, W)

    despeckled = np.full((H, W), 0.5, np.float32)
    recon = despeckled.copy()
    recon[10:30, 10:30] = 0.9  # lesion the autoencoder could not reconstruct

    mse, visual, mean_mse = engine.compute_anomaly(frame_id, despeckled, recon)
    assert mse.dtype == np.float32 and mse.shape == (H, W)
    assert mse[15:25, 15:25].mean() > 0.1
    assert mse[40:60, 40:60].max() < 1e-6
    assert visual.shape == (H, W, 3)
    assert mean_mse > 0


def test_identical_inputs_give_zero_anomaly(engine):
    frame_id, _ = next_frame(engine)
    img = np.random.default_rng(0).random((H, W)).astype(np.float32)
    mse, _, mean_mse = engine.compute_anomaly(frame_id, img, img.copy())
    assert mean_mse == 0.0 and mse.max() == 0.0


def test_returned_arrays_survive_next_frame(engine):
    frame_id, raw1 = next_frame(engine)
    raw1_snapshot = raw1.copy()
    a = np.full((H, W), 0.5, np.float32)
    b = a.copy()
    b[:32] = 1.0
    mse1, vis1, _ = engine.compute_anomaly(frame_id, a, b)
    mse1_snapshot, vis1_snapshot = mse1.copy(), vis1.copy()

    frame_id2, _ = next_frame(engine)
    mse2, vis2, _ = engine.compute_anomaly(frame_id2, a, a.copy())

    assert np.array_equal(mse1, mse1_snapshot)
    assert np.array_equal(vis1, vis1_snapshot)
    assert np.array_equal(raw1, raw1_snapshot)
    assert mse2.max() == 0.0


def test_stale_or_reused_frame_id_is_rejected(engine):
    img = np.zeros((H, W), np.float32)
    with pytest.raises(ValueError):
        engine.compute_anomaly(1, img, img)  # nothing pending yet

    old_id, _ = next_frame(engine)
    new_id, _ = next_frame(engine)
    assert new_id != old_id
    with pytest.raises(ValueError):
        engine.compute_anomaly(old_id, img, img)  # superseded frame

    engine.compute_anomaly(new_id, img, img)
    with pytest.raises(ValueError):
        engine.compute_anomaly(new_id, img, img)  # already consumed


def test_non_contiguous_and_bad_shape_inputs(engine):
    frame_id, _ = next_frame(engine)
    base = np.random.default_rng(1).random((W, H)).astype(np.float32)
    despeckled = base.T  # non-contiguous view, shape (H, W)
    assert not despeckled.flags["C_CONTIGUOUS"]
    mse, _, mean_mse = engine.compute_anomaly(frame_id, despeckled, np.ascontiguousarray(despeckled))
    assert mean_mse == 0.0 and mse.max() == 0.0

    frame_id2, _ = next_frame(engine)
    with pytest.raises(ValueError):
        engine.compute_anomaly(frame_id2, np.zeros((H, W, 1), np.float32), np.zeros((H, W), np.float32))
