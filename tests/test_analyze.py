"""
tests/test_analyze.py
=====================
Tests for statistical analysis pipeline: paired bootstrap CI, Holm-Bonferroni step-down,
and results aggregation table formatting.
"""

import os
import csv
import numpy as np
import pytest

from experiments.analyze import (
    compute_paired_bootstrap_ci,
    holm_bonferroni_correction,
    read_results_csv,
    generate_summary_table,
)


def test_paired_bootstrap_ci_positive_shift():
    baseline = np.array([0.80, 0.81, 0.82])
    candidate = np.array([0.85, 0.86, 0.87])
    mean_diff, ci_l, ci_u, p_val = compute_paired_bootstrap_ci(baseline, candidate, n_bootstraps=500, seed=42)
    assert np.isclose(mean_diff, 0.05, atol=1e-4)
    assert ci_l > 0.0
    assert ci_u > ci_l
    assert p_val <= 0.05


def test_paired_bootstrap_ci_zero_shift():
    baseline = np.array([0.80, 0.82, 0.81])
    candidate = np.array([0.80, 0.82, 0.81])
    mean_diff, ci_l, ci_u, p_val = compute_paired_bootstrap_ci(baseline, candidate, n_bootstraps=500, seed=42)
    assert np.isclose(mean_diff, 0.0, atol=1e-4)
    assert ci_l <= 0.0 <= ci_u


def test_holm_bonferroni_correction_ordering_and_multipliers():
    # 4 p-values: 0.01, 0.04, 0.02, 0.10
    # Sorted: p1=0.01 (x4 -> 0.04), p2=0.02 (x3 -> 0.06), p3=0.04 (x2 -> 0.08), p4=0.10 (x1 -> 0.10)
    p_vals = [0.01, 0.04, 0.02, 0.10]
    corrected = holm_bonferroni_correction(p_vals, alpha=0.05)
    
    assert len(corrected) == 4
    # Check that original indices are preserved
    p_adj_0, sig_0 = corrected[0]
    p_adj_1, sig_1 = corrected[1]
    p_adj_2, sig_2 = corrected[2]
    p_adj_3, sig_3 = corrected[3]

    assert np.isclose(p_adj_0, 0.04)
    assert sig_0 is True  # 0.04 < 0.05
    assert np.isclose(p_adj_2, 0.06)
    assert sig_2 is False # 0.06 >= 0.05
    assert np.isclose(p_adj_1, 0.08)
    assert sig_1 is False
    assert np.isclose(p_adj_3, 0.10)
    assert sig_3 is False


def test_results_csv_aggregation_and_summary_table(tmp_path):
    csv_file = tmp_path / "results.csv"
    fieldnames = ["run_id", "git_sha", "front_end", "family", "seed", "split", "metric", "value"]
    
    rows = [
        {"run_id": "r1", "git_sha": "abc", "front_end": "raw", "family": "b", "seed": 1, "split": "test", "metric": "macro_auroc", "value": 0.80},
        {"run_id": "r2", "git_sha": "abc", "front_end": "raw", "family": "b", "seed": 2, "split": "test", "metric": "macro_auroc", "value": 0.81},
        {"run_id": "r3", "git_sha": "abc", "front_end": "raw", "family": "b", "seed": 3, "split": "test", "metric": "macro_auroc", "value": 0.82},
        {"run_id": "r4", "git_sha": "abc", "front_end": "unet_synth", "family": "b", "seed": 1, "split": "test", "metric": "macro_auroc", "value": 0.86},
        {"run_id": "r5", "git_sha": "abc", "front_end": "unet_synth", "family": "b", "seed": 2, "split": "test", "metric": "macro_auroc", "value": 0.87},
        {"run_id": "r6", "git_sha": "abc", "front_end": "unet_synth", "family": "b", "seed": 3, "split": "test", "metric": "macro_auroc", "value": 0.88},
    ]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    records = read_results_csv(str(csv_file))
    assert len(records) == 6

    table = generate_summary_table(records, family="b", split="test", metric="macro_auroc")
    assert "Family B" in table
    assert "`raw` (Baseline)" in table
    assert "`unet_synth`" in table
    assert "+0.0600" in table
