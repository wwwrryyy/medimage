"""
roi_logic.py
============
Anomaly ROI Decision Logic & Spatial Non-Maximum Suppression (NMS).

Single owner of anomaly thresholding, morphological noise suppression, contour
extraction, Non-Maximum Suppression, and temperature-scaled confidence calibration
over C++ anomaly heatmaps.
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


class ROIDecisionLogic:
    """
    Filters spatial MSE anomaly heatmaps, runs contour clustering, applies NMS,
    and produces calibrated finding confidence scores.

    Args:
        mse_threshold: Pixel-wise MSE cutoff above which reconstruction error is flagged.
        min_area: Minimum contour pixel area to retain (filters isolated speckle).
        max_area: Maximum contour pixel area to retain.
        nms_iou_threshold: IoU threshold for Non-Maximum Suppression.
        temperature: Temperature parameter (> 0) for logistic confidence calibration
            over normalized excess MSE ((score - mse_threshold) / mse_threshold).
    """

    def __init__(
        self,
        mse_threshold: float = 0.06,
        min_area: int = 150,
        max_area: int = 15000,
        nms_iou_threshold: float = 0.3,
        temperature: float = 0.5,
    ):
        if mse_threshold <= 0.0:
            raise ValueError("mse_threshold must be > 0")
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        self.mse_threshold = float(mse_threshold)
        self.min_area = int(min_area)
        self.max_area = int(max_area)
        self.nms_iou_threshold = float(nms_iou_threshold)
        self.temperature = float(temperature)

    def set_mse_threshold(self, th: float) -> None:
        if th <= 0.0:
            raise ValueError("mse_threshold must be > 0")
        self.mse_threshold = float(th)

    def calibrate_confidence(self, score: float) -> float:
        """
        Maps an ROI's mean MSE score to a calibrated confidence in (0.0, 1.0)
        using a temperature-scaled logistic curve anchored at mse_threshold:
            margin = (score - mse_threshold) / mse_threshold
            confidence = 1 / (1 + exp(-margin / temperature))
        At score == mse_threshold, confidence is 0.5; higher MSE saturates smoothly toward 1.0.
        """
        margin = (float(score) - self.mse_threshold) / self.mse_threshold
        z = max(-60.0, min(60.0, margin / self.temperature))
        return float(1.0 / (1.0 + math.exp(-z)))

    def compute_iou(self, boxA: Sequence[int], boxB: Sequence[int]) -> float:
        """Calculates Intersection over Union (IoU) between two bounding boxes [x, y, w, h]."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return float(iou)

    def apply_nms(
        self, boxes: List[List[int]], scores: List[float]
    ) -> Tuple[List[List[int]], List[float]]:
        """Performs Non-Maximum Suppression (NMS) over bounding boxes."""
        if len(boxes) == 0:
            return [], []

        boxes_arr = np.array(boxes)
        scores_arr = np.array(scores)

        # Sort boxes by anomaly score descending
        idxs = np.argsort(scores_arr)[::-1]
        keep = []

        while len(idxs) > 0:
            current = idxs[0]
            keep.append(current)

            if len(idxs) == 1:
                break

            rest = idxs[1:]
            filtered_idxs = []
            for i in rest:
                iou = self.compute_iou(boxes_arr[current], boxes_arr[i])
                if iou < self.nms_iou_threshold:
                    filtered_idxs.append(i)
            idxs = np.array(filtered_idxs)

        keep_boxes = [boxes[i] for i in keep]
        keep_scores = [scores[i] for i in keep]
        return keep_boxes, keep_scores

    def detect_anomaly_rois(
        self, mse_heatmap: Optional[np.ndarray]
    ) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray]]:
        """
        Processes C++ MSE anomaly matrix and returns consolidated anomaly ROIs.

        Args:
            mse_heatmap (np.ndarray): Pixel-wise MSE matrix (float32, 0.0 - 1.0)

        Returns:
            rois (list): List of dicts [{'id', 'bbox', 'score', 'type', 'confidence'}]
            binary_mask (np.ndarray): Binary thresholded and morphologically cleaned anomaly mask
        """
        if mse_heatmap is None or mse_heatmap.size == 0:
            return [], None

        # 1. Thresholding owned solely by ROIDecisionLogic
        binary_mask = (mse_heatmap > self.mse_threshold).astype(np.uint8) * 255

        # 2. Morphological filtering to discard transient visual noise (isolated speckle)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

        # 3. Contour Detection & Finding Extraction
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        finding_boxes: List[List[int]] = []
        finding_scores: List[float] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_area <= area <= self.max_area:
                x, y, w, h = cv2.boundingRect(cnt)

                # Compute localized mean anomaly intensity score over active anomaly pixels in the ROI
                roi_mse = mse_heatmap[y : y + h, x : x + w]
                active = roi_mse[roi_mse > self.mse_threshold]
                if active.size > 0:
                    score = float(np.mean(active))
                elif roi_mse.size > 0:
                    score = float(np.mean(roi_mse))
                else:
                    score = float(self.mse_threshold)

                finding_boxes.append([x, y, w, h])
                finding_scores.append(score)

        # 4. Apply Spatial Non-Maximum Suppression (NMS)
        nms_boxes, nms_scores = self.apply_nms(finding_boxes, finding_scores)

        # 5. Format anomaly ROI findings
        rois: List[Dict[str, Any]] = []
        for i, (box, score) in enumerate(zip(nms_boxes, nms_scores)):
            x, y, w, h = box

            # Region size category based on bounding-box area
            area = w * h
            if area > 1200:
                classification = "LARGE ANOMALY REGION"
            elif area > 400:
                classification = "MEDIUM ANOMALY REGION"
            else:
                classification = "SMALL ANOMALY REGION"

            rois.append(
                {
                    "id": i + 1,
                    "bbox": [x, y, w, h],
                    "score": score,
                    "type": classification,
                    "confidence": self.calibrate_confidence(score),
                }
            )

        return rois, cleaned_mask
