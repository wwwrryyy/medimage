"""
models/classifier/conformal.py
==============================
Split conformal prediction and distribution-free referral calibration for retinal OCT diagnostics.

Provides:
1. Multi-class conformal prediction sets with finite-sample coverage guarantees (1 - alpha).
2. Triage referral thresholding guaranteeing >= 1 - beta sensitivity for pathological findings.
"""

from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np


class ConformalPredictor:
    """
    Split conformal prediction for multi-class classification.
    Guarantees that P(Y in C(X)) >= 1 - alpha for exchangeable test data.
    """

    def __init__(self, alpha: float = 0.05):
        """
        Args:
            alpha: Significance level (e.g. 0.05 for 95% coverage guarantee).
        """
        self.alpha = float(alpha)
        self.quantile_threshold: Optional[float] = None
        self.calibrated = False

    def calibrate(self, val_probs: np.ndarray, val_labels: np.ndarray) -> float:
        """
        Computes the empirical quantile threshold on calibration (validation) set.
        Args:
            val_probs: Array of predicted softmax probabilities [N, num_classes].
            val_labels: Ground-truth integer labels [N].
        Returns:
            hat_q: Fitted quantile threshold.
        """
        n = len(val_labels)
        if n == 0:
            raise ValueError("Validation set cannot be empty for calibration.")

        val_probs = np.asarray(val_probs, dtype=np.float64)
        val_labels = np.asarray(val_labels, dtype=np.int64)

        # Non-conformity score s_i = 1 - P(Y_i | X_i)
        true_class_probs = val_probs[np.arange(n), val_labels]
        scores = 1.0 - true_class_probs

        # Conformal quantile with finite-sample correction ceil((n + 1)(1 - alpha)) / n
        q_level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
        q_level = min(1.0, max(0.0, q_level))

        self.quantile_threshold = float(np.quantile(scores, q_level, method="higher"))
        self.calibrated = True
        return self.quantile_threshold

    def predict_set(self, test_probs: np.ndarray) -> List[List[int]]:
        """
        Constructs prediction set C(X) = {y: P(y | X) >= 1 - hat_q}.
        Args:
            test_probs: Predicted probabilities [M, num_classes].
        Returns:
            List of class index lists for each test sample.
        """
        if not self.calibrated or self.quantile_threshold is None:
            raise RuntimeError("ConformalPredictor must be calibrated before calling predict_set.")

        test_probs = np.asarray(test_probs, dtype=np.float64)
        p_cutoff = max(0.0, 1.0 - self.quantile_threshold)

        prediction_sets: List[List[int]] = []
        for p in test_probs:
            selected = np.where(p >= p_cutoff)[0].tolist()
            if not selected:
                # If no class reaches threshold, take argmax to prevent empty set
                selected = [int(np.argmax(p))]
            prediction_sets.append(selected)
        return prediction_sets

    def evaluate_coverage(self, test_probs: np.ndarray, test_labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluates empirical coverage and average prediction set size on test data.
        """
        pred_sets = self.predict_set(test_probs)
        test_labels = np.asarray(test_labels, dtype=np.int64)

        covered = sum(y in p_set for y, p_set in zip(test_labels, pred_sets))
        empirical_coverage = covered / len(test_labels) if len(test_labels) > 0 else 0.0
        avg_set_size = float(np.mean([len(s) for s in pred_sets])) if pred_sets else 0.0

        return {
            "target_coverage": 1.0 - self.alpha,
            "empirical_coverage": empirical_coverage,
            "avg_set_size": avg_set_size,
            "quantile_threshold": self.quantile_threshold,
        }


class ReferralCalibrator:
    """
    Calibrates point-of-care referral thresholds with guaranteed sensitivity.
    Assumes normal class index is 3 (CNV=0, DME=1, DRUSEN=2, NORMAL=3).
    """

    def __init__(self, normal_class_idx: int = 3, target_sensitivity: float = 0.95):
        """
        Args:
            normal_class_idx: Index of healthy / normal tissue class.
            target_sensitivity: Minimum desired sensitivity for referring true pathologies (e.g. 0.95).
        """
        self.normal_class_idx = normal_class_idx
        self.target_sensitivity = float(target_sensitivity)
        self.referral_prob_threshold: Optional[float] = None
        self.calibrated = False

    def calibrate(self, val_probs: np.ndarray, val_labels: np.ndarray) -> float:
        """
        Fits minimum pathology probability threshold to guarantee target sensitivity.
        """
        val_probs = np.asarray(val_probs, dtype=np.float64)
        val_labels = np.asarray(val_labels, dtype=np.int64)

        # Pathological scans: label != normal_class_idx
        path_mask = val_labels != self.normal_class_idx
        path_probs = 1.0 - val_probs[path_mask, self.normal_class_idx]
        n_path = len(path_probs)

        if n_path == 0:
            raise ValueError("Validation set contains no pathological samples for referral calibration.")

        # Non-conformity on pathological cases: score = 1 - P(pathology)
        scores = 1.0 - path_probs
        q_level = np.ceil((n_path + 1) * self.target_sensitivity) / n_path
        q_level = min(1.0, max(0.0, q_level))

        hat_q = float(np.quantile(scores, q_level, method="higher"))
        self.referral_prob_threshold = max(0.01, min(0.99, 1.0 - hat_q))
        self.calibrated = True
        return self.referral_prob_threshold

    def triage(self, probs: np.ndarray) -> Dict[str, Union[bool, str, float]]:
        """
        Triages an individual scan into:
        - "REFER" (Pathology detected with statistical guarantee)
        - "NO_REFERRAL" (Verified healthy tissue)
        - "REFER_INDETERMINATE" (Ambiguous/uncertain prediction)
        """
        if not self.calibrated or self.referral_prob_threshold is None:
            raise RuntimeError("ReferralCalibrator must be calibrated before triage.")

        probs = np.asarray(probs, dtype=np.float64)
        p_pathology = float(1.0 - probs[self.normal_class_idx])
        pred_class = int(np.argmax(probs))

        if p_pathology >= self.referral_prob_threshold:
            decision = "REFER"
            needs_referral = True
        elif pred_class == self.normal_class_idx and probs[self.normal_class_idx] >= 0.85:
            decision = "NO_REFERRAL"
            needs_referral = False
        else:
            decision = "REFER_INDETERMINATE"
            needs_referral = True

        return {
            "decision": decision,
            "needs_referral": needs_referral,
            "pathology_probability": p_pathology,
            "threshold": self.referral_prob_threshold,
        }
