"""End-to-end tests for ScanSession across both fake in-memory and ONNX adapters."""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src_python"))
pytest.importorskip("scan_core", reason="scan_core not built")

from scan_session import ScanSession
from sharp_resilience import SHARPOfflineResilienceManager

H, W = 128, 128


def _make_test_video(tmp_path, frames=8, height=H, width=W) -> str:
    video = str(tmp_path / "session_scan.avi")
    writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (width, height), False)
    for _ in range(frames):
        img = np.full((height, width), 128, np.uint8)
        # Inject bright anomalous lesion region [30:70, 40:80] (40x40 = 1600 px > 1200 px)
        img[30:70, 40:80] = 240
        writer.write(img)
    writer.release()
    return video


def _pull_result(session: ScanSession, retries: int = 100):
    for _ in range(retries):
        res = session.process_next()
        if res is not None:
            return res
    pytest.fail("ScanSession did not yield a FrameResult in time")


def test_session_with_fake_adapters_detects_finding_and_logs(tmp_path):
    video = _make_test_video(tmp_path)
    logged = []

    # Fake despeckler: identity; fake reconstructor: healthy flat baseline (0.5)
    despeckler = lambda img: img
    reconstructor = lambda img: np.full_like(img, 0.5, dtype=np.float32)

    with ScanSession(
        source=video,
        despeckler=despeckler,
        reconstructor=reconstructor,
        on_finding=logged.append,
    ) as session:
        res = _pull_result(session)
        assert res.frame_id >= 1
        assert res.despeckled_frame.shape == (H, W)
        assert res.reconstruction.shape == (H, W)
        assert res.anomaly_heatmap.shape == (H, W)
        assert res.visual_heatmap.shape == (H, W, 3)
        assert res.mean_mse > 0.01
        assert len(res.findings) == 1
        assert res.findings[0]["type"] == "LARGE ANOMALY REGION"
        assert len(logged) == 1
        assert any(s in session.health_status for s in ("NOMINAL", "ADAPTIVE", "SELF-HEALING"))


def test_session_with_offline_resilience_callback(tmp_path):
    video = _make_test_video(tmp_path)
    db_path = str(tmp_path / "telemetry.db")
    resilience = SHARPOfflineResilienceManager(db_path=db_path)
    try:
        with ScanSession(
            source=video,
            despeckler=lambda img: img,
            reconstructor=lambda img: np.full_like(img, 0.5, dtype=np.float32),
            on_finding=resilience.log_anomaly_offline,
        ) as session:
            res = _pull_result(session)
            assert len(res.findings) >= 1

        assert resilience.get_pending_sync_count() == len(res.findings)
        recent = resilience.get_recent_logs(limit=5)
        assert recent[0]["type"] == "LARGE ANOMALY REGION"
    finally:
        resilience.auto_sync_running = False


@pytest.mark.skipif(
    not (
        os.path.exists("exports/unet_despeckler.onnx")
        and os.path.exists("exports/conv_autoencoder_tissue_baseline.onnx")
    ),
    reason="ONNX exports not present",
)
def test_session_from_onnx_handles_arbitrary_frame_resolution(tmp_path):
    # Use 96x140 (neither 256x256 nor a multiple of 4 in both dims if we pick 95x141)
    video = _make_test_video(tmp_path, height=96, width=140)
    with ScanSession.from_onnx(source=video) as session:
        res = _pull_result(session)
        assert res.despeckled_frame.shape == (96, 140)
        assert res.reconstruction.shape == (96, 140)
        assert res.anomaly_heatmap.shape == (96, 140)
        assert np.isfinite(res.mean_mse)
