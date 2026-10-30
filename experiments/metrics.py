"""
experiments/metrics.py
======================
Standardized, standalone evaluation metrics pre-registered in EXPERIMENT_PLAN.md:
1. Macro AUROC (One-vs-Rest via Wilcoxon-Mann-Whitney rank statistic)
2. Expected Calibration Error (ECE, 15 uniform confidence bins)
3. Area Under Per-Region Overlap (AUPRO up to max FPR, e.g. 0.3)
4. Dice Similarity Coefficient at threshold
"""

from typing import List, Tuple, Union
import numpy as np


def compute_binary_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Computes exact Area Under the ROC Curve for binary classification
    using the Wilcoxon-Mann-Whitney rank statistic:
        AUC = (R_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    """
    y_true = np.asarray(y_true, dtype=bool)
    y_score = np.asarray(y_score, dtype=np.float64)

    n_pos = int(np.sum(y_true))
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)

    # Handle ties with average ranks
    _, unique_inv, counts = np.unique(y_score, return_inverse=True, return_counts=True)
    tie_indices = np.where(counts > 1)[0]
    if len(tie_indices) > 0:
        for idx in tie_indices:
            mask = (unique_inv == idx)
            ranks[mask] = np.mean(ranks[mask])

    rank_sum_pos = np.sum(ranks[y_true])
    u_stat = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
    return float(u_stat / (n_pos * n_neg))


def compute_macro_auroc(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int = 4) -> float:
    """
    Computes Macro AUROC (unweighted mean of One-vs-Rest AUROCs across classes).
    y_true: (N,) integer class indices [0..num_classes-1]
    y_prob: (N, num_classes) predicted class probabilities
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    aucs = []
    for c in range(num_classes):
        bin_true = (y_true == c)
        auc = compute_binary_auroc(bin_true, y_prob[:, c])
        if not np.isnan(auc):
            aucs.append(auc)

    return float(np.mean(aucs)) if aucs else float("nan")


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """
    Computes Expected Calibration Error (ECE) with uniform confidence binning.
    y_true: (N,) integer labels
    y_prob: (N, C) probability distributions
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    confidences = np.max(y_prob, axis=-1)
    predictions = np.argmax(y_prob, axis=-1)
    accuracies = (predictions == y_true)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)

    if n_samples == 0:
        return 0.0

    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > low) & (confidences <= high) if i > 0 else (confidences >= low) & (confidences <= high)
        bin_count = int(np.sum(mask))

        if bin_count > 0:
            bin_acc = float(np.mean(accuracies[mask]))
            bin_conf = float(np.mean(confidences[mask]))
            ece += (bin_count / n_samples) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_dice_score(gt_mask: np.ndarray, pred_mask: np.ndarray, eps: float = 1e-6) -> float:
    """Computes Dice similarity coefficient between binary ground truth and prediction."""
    gt = np.asarray(gt_mask, dtype=bool)
    pred = np.asarray(pred_mask, dtype=bool)
    intersection = np.logical_and(gt, pred).sum()
    return float((2.0 * intersection + eps) / (gt.sum() + pred.sum() + eps))


def compute_aupro(
    gt_masks: List[np.ndarray],
    pred_heatmaps: List[np.ndarray],
    max_fpr: float = 0.30,
    num_thresholds: int = 50,
) -> float:
    """
    Computes Area Under the Per-Region Overlap curve (AUPRO) up to max_fpr.
    Evaluates how thoroughly anomalous regions (connected components) are covered
    across candidate anomaly thresholds.
    """
    import cv2

    assert len(gt_masks) == len(pred_heatmaps), "Mismatch between masks and heatmaps"
    if not gt_masks:
        return 0.0

    # Extract connected components (regions) for all ground truth masks
    regions = []  # list of (mask_idx, component_mask, component_area)
    total_negative_pixels = 0

    for idx, gt in enumerate(gt_masks):
        gt_bin = (gt > 0).astype(np.uint8)
        neg_count = (gt_bin == 0).sum()
        total_negative_pixels += neg_count

        num_labels, labels = cv2.connectedComponents(gt_bin)
        for label_id in range(1, num_labels):
            comp_mask = (labels == label_id)
            comp_area = int(comp_mask.sum())
            if comp_area > 0:
                regions.append((idx, comp_mask, comp_area))

    if not regions or total_negative_pixels == 0:
        return float("nan")

    # Sample thresholds between min and max heatmap values
    all_scores = np.concatenate([h.flatten() for h in pred_heatmaps])
    thresholds = np.linspace(all_scores.min(), all_scores.max(), num_thresholds)
    thresholds = np.sort(thresholds)[::-1]  # high to low threshold

    fpr_list = []
    pro_list = []

    for th in thresholds:
        # Compute False Positive Rate (FPR) across healthy pixels
        false_positives = 0
        for idx, hm in enumerate(pred_heatmaps):
            pred_bin = hm >= th
            gt_bin = gt_masks[idx] > 0
            false_positives += np.logical_and(pred_bin, ~gt_bin).sum()

        fpr = false_positives / total_negative_pixels
        if fpr > max_fpr:
            break

        # Compute Per-Region Overlap (PRO): mean overlap ratio across all regions
        overlaps = []
        for idx, comp_mask, comp_area in regions:
            pred_bin = pred_heatmaps[idx] >= th
            overlap = np.logical_and(pred_bin, comp_mask).sum() / comp_area
            overlaps.append(overlap)

        pro = float(np.mean(overlaps)) if overlaps else 0.0
        fpr_list.append(fpr)
        pro_list.append(pro)

    if not fpr_list:
        return 0.0

    # Normalize AUPRO area by max_fpr
    fpr_arr = np.array(fpr_list)
    pro_arr = np.array(pro_list)

    # Sort by FPR for trapezoidal integration
    sort_idx = np.argsort(fpr_arr)
    fpr_sorted = list(fpr_arr[sort_idx])
    pro_sorted = list(pro_arr[sort_idx])

    if fpr_sorted[0] > 0.0:
        fpr_sorted.insert(0, 0.0)
        pro_sorted.insert(0, 0.0)

    # Extend to max_fpr if the highest observed FPR is below max_fpr
    if fpr_sorted[-1] < max_fpr:
        fpr_sorted.append(max_fpr)
        pro_sorted.append(pro_sorted[-1])

    # Integrate up to max_fpr
    aupro_val = float(np.trapz(pro_sorted, fpr_sorted) / max_fpr)
    return min(1.0, max(0.0, aupro_val))
