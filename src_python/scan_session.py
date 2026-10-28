"""
scan_session.py
===============
Deep module owning the end-to-end "Frame -> Findings" diagnostic pipeline:
    Raw Frame -> Despeckled Frame -> Reconstruction -> Anomaly Heatmap -> Findings
    (-> optional Stage 3 pathology classification of each Finding's 64x64 ROI crop)

Hides frame_id sequencing, grayscale/float32 normalization, model input resolution
adaptation, and optional offline telemetry callbacks behind a single interface.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import cv2
import numpy as np

try:
    import scan_core
except ImportError:
    from src_python import scan_core

try:
    from roi_logic import ROIDecisionLogic
except ImportError:
    from src_python.roi_logic import ROIDecisionLogic


try:
    from onnx_model_io import OnnxCropClassifierAdapter, OnnxImageAdapter
    from pathology import crop_roi, describe_pathology
except ImportError:
    from src_python.onnx_model_io import OnnxCropClassifierAdapter, OnnxImageAdapter
    from src_python.pathology import crop_roi, describe_pathology


class ImageToImageAdapter(Protocol):
    """Seam for 2D float32 [H, W] in [0.0, 1.0] -> 2D float32 [H, W] in [0.0, 1.0] inference."""

    def __call__(self, image: np.ndarray) -> np.ndarray:
        ...


class CropClassifier(Protocol):
    """Seam for 2D float32 ROI crop [H, W] in [0.0, 1.0] -> 1D class-probability vector [C]."""

    def __call__(self, crop: np.ndarray) -> np.ndarray:
        ...


@dataclass(frozen=True)
class FrameResult:
    """Consolidated output for a single processed frame in a ScanSession."""

    frame_id: int
    raw_frame: np.ndarray
    despeckled_frame: np.ndarray
    reconstruction: np.ndarray
    anomaly_heatmap: np.ndarray
    visual_heatmap: np.ndarray
    mean_mse: float
    findings: List[Dict[str, Any]]
    anomaly_mask: Optional[np.ndarray]


def _to_unit_grayscale(raw: np.ndarray) -> np.ndarray:
    """Converts a raw uint8/float frame (2D or 3D BGR) to 2D float32 in [0.0, 1.0]."""
    if raw.ndim == 3 and raw.shape[2] == 3:
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    elif raw.ndim == 3 and raw.shape[2] == 1:
        gray = raw[:, :, 0]
    elif raw.ndim == 2:
        gray = raw
    else:
        raise ValueError(f"Unsupported raw frame shape: {raw.shape}")

    if gray.dtype == np.uint8:
        return gray.astype(np.float32) / 255.0
    return np.clip(gray.astype(np.float32), 0.0, 1.0)


class ScanSession:
    """
    Deep module coordinating video ingestion, speckle suppression, healthy-tissue
    reconstruction, C++ pixel-wise MSE anomaly calculus, and ROI finding extraction.
    """

    def __init__(
        self,
        source: str,
        despeckler: ImageToImageAdapter,
        reconstructor: ImageToImageAdapter,
        roi_logic: Optional[ROIDecisionLogic] = None,
        on_finding: Optional[Callable[[Dict[str, Any]], None]] = None,
        engine: Optional[Any] = None,
        classifier: Optional[CropClassifier] = None,
    ):
        self.source = str(source)
        self.despeckler = despeckler
        self.reconstructor = reconstructor
        self.roi_logic = roi_logic or ROIDecisionLogic()
        self.on_finding = on_finding
        self.classifier = classifier
        self._engine = engine or scan_core.ScanEngine()
        self._started = False

    @classmethod
    def from_onnx(
        cls,
        source: str,
        despeckler_onnx: str = "exports/unet_despeckler.onnx",
        reconstructor_onnx: str = "exports/conv_autoencoder_tissue_baseline.onnx",
        roi_logic: Optional[ROIDecisionLogic] = None,
        on_finding: Optional[Callable[[Dict[str, Any]], None]] = None,
        classifier_onnx: Optional[str] = None,
    ) -> "ScanSession":
        """
        Constructs a ScanSession backed by ONNX Runtime CPU adapters.
        If classifier_onnx is given, each finding is also classified (Stage 3).
        """
        return cls(
            source=source,
            despeckler=OnnxImageAdapter(despeckler_onnx),
            reconstructor=OnnxImageAdapter(reconstructor_onnx),
            roi_logic=roi_logic,
            on_finding=on_finding,
            classifier=OnnxCropClassifierAdapter(classifier_onnx) if classifier_onnx else None,
        )

    def start(self) -> "ScanSession":
        if not self._started:
            self._engine.start_stream(self.source)
            self._started = True
        return self

    def stop(self) -> None:
        if self._started:
            self._engine.stop_stream()
            self._started = False

    def __enter__(self) -> "ScanSession":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    def process_next(self) -> Optional[FrameResult]:
        """
        Pops the next frame from the stream, runs Stage 1 despeckling and Stage 2
        reconstruction + C++ anomaly heatmap + ROI clustering, and invokes on_finding
        for each detected finding. Returns None if no frame is currently available.
        """
        if not self._started:
            self.start()

        has_frame, frame_id, raw = self._engine.next_frame()
        if not has_frame or raw is None:
            return None

        gray_f32 = _to_unit_grayscale(raw)
        despeckled = np.ascontiguousarray(self.despeckler(gray_f32), dtype=np.float32)
        reconstruction = np.ascontiguousarray(self.reconstructor(despeckled), dtype=np.float32)

        mse_heatmap, visual_heatmap, mean_mse = self._engine.compute_anomaly(
            frame_id, despeckled, reconstruction
        )
        findings, binary_mask = self.roi_logic.detect_anomaly_rois(mse_heatmap)

        # Stage 3: classify each finding's 64x64 ROI crop (taken from the despeckled frame).
        if self.classifier is not None:
            for finding in findings:
                crop = crop_roi(despeckled, finding["bbox"])
                finding["pathology"] = describe_pathology(self.classifier(crop))

        if self.on_finding is not None:
            for finding in findings:
                self.on_finding(finding)

        return FrameResult(
            frame_id=int(frame_id),
            raw_frame=raw,
            despeckled_frame=despeckled,
            reconstruction=reconstruction,
            anomaly_heatmap=mse_heatmap,
            visual_heatmap=visual_heatmap,
            mean_mse=float(mean_mse),
            findings=findings,
            anomaly_mask=binary_mask,
        )

    @property
    def health_status(self) -> str:
        return self._engine.get_health_status()

    @property
    def fps(self) -> float:
        return float(self._engine.get_fps())

    @property
    def queue_size(self) -> int:
        return int(self._engine.get_queue_size())
