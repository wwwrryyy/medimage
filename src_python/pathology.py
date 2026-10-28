"""
pathology.py
============
Torch-free helpers shared by Stage 3 training and the runtime ScanSession:
pathology class labels, square ROI crop extraction, and probability formatting.
"""

from typing import Any, Dict, Sequence

import cv2
import numpy as np

# Kermany OCT diagnostic classes; order defines the classifier output index.
PATHOLOGY_CLASSES = ("CNV", "DME", "DRUSEN", "NORMAL")

# Fixed classifier input resolution (PRD Stage 3: 64x64 crops).
ROI_CROP_SIZE = 64


def crop_roi(frame: np.ndarray, bbox: Sequence[int], size: int = ROI_CROP_SIZE) -> np.ndarray:
    """
    Extracts a square crop around a finding bounding box [x, y, w, h] and resizes it
    to (size, size). The box is expanded to a square (side = max(w, h)) centred on the
    finding and shifted to stay inside the frame, so aspect ratio is never distorted.

    Args:
        frame: 2D float32 frame [H, W] (the despeckled frame).
        bbox: [x, y, w, h] in frame pixel coordinates.
    Returns:
        float32 array of shape (size, size).
    """
    if frame.ndim != 2:
        raise ValueError(f"Expected 2D frame [H, W], got shape {frame.shape}")

    x, y, w, h = (int(v) for v in bbox)
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid bbox {list(bbox)}: width and height must be > 0")

    frame_h, frame_w = frame.shape
    side = min(max(w, h), frame_h, frame_w)
    cx, cy = x + w // 2, y + h // 2
    x0 = int(np.clip(cx - side // 2, 0, frame_w - side))
    y0 = int(np.clip(cy - side // 2, 0, frame_h - side))

    crop = frame[y0:y0 + side, x0:x0 + side].astype(np.float32, copy=False)
    interp = cv2.INTER_AREA if side > size else cv2.INTER_LINEAR
    return cv2.resize(crop, (size, size), interpolation=interp).astype(np.float32)


def describe_pathology(
    probabilities: Sequence[float],
    class_names: Sequence[str] = PATHOLOGY_CLASSES,
) -> Dict[str, Any]:
    """Formats a class-probability vector as {'label', 'confidence', 'probabilities'}."""
    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if probs.shape[0] != len(class_names):
        raise ValueError(
            f"Classifier returned {probs.shape[0]} probabilities, expected {len(class_names)}"
        )
    best = int(np.argmax(probs))
    return {
        "label": class_names[best],
        "confidence": float(probs[best]),
        "probabilities": {name: float(p) for name, p in zip(class_names, probs)},
    }
