"""
experiments/analyze.py
======================
Statistical analysis and report generator pre-registered in EXPERIMENT_PLAN.md:
1. Aggregates mean ± std across 3 seeds from `experiments/results.csv`.
2. Computes paired bootstrap 95% confidence intervals (2,000 resamples) for front-end comparisons vs. `raw`.
3. Applies Holm-Bonferroni step-down correction across the 4 non-raw front-ends within each family.
4. Generates formatted summary tables.

Schema of experiments/results.csv:
    run_id, git_sha, front_end, family, seed, split, metric, value
"""

import os
import sys
import csv
import argparse
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict

import numpy as np


def compute_paired_bootstrap_ci(
    baseline_values: np.ndarray,
    candidate_values: np.ndarray,
    n_bootstraps: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float, float]:
    """
    Computes paired difference bootstrap confidence interval and empirical two-tailed p-value:
        diff = candidate - baseline
    Returns:
        (mean_diff, ci_lower, ci_upper, p_value)
    """
    baseline_values = np.asarray(baseline_values, dtype=np.float64)
    candidate_values = np.asarray(candidate_values, dtype=np.float64)

    if len(baseline_values) != len(candidate_values):
        raise ValueError("baseline and candidate values must have matching length for paired bootstrap")
    if len(baseline_values) == 0:
        return 0.0, 0.0, 0.0, 1.0

    diffs = candidate_values - baseline_values
    mean_diff = float(np.mean(diffs))

    rng = np.random.default_rng(seed)
    n_samples = len(diffs)
    # Resample paired differences with replacement
    boot_indices = rng.integers(0, n_samples, size=(n_bootstraps, n_samples))
    boot_means = np.mean(diffs[boot_indices], axis=1)

    ci_lower = float(np.percentile(boot_means, 100 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_means, 100 * (1.0 - alpha / 2.0)))

    # Two-sided empirical p-value tested against null hypothesis (mean_diff == 0)
    frac_le_0 = np.mean(boot_means <= 0.0)
    frac_ge_0 = np.mean(boot_means >= 0.0)
    p_val = float(min(1.0, 2.0 * min(frac_le_0, frac_ge_0)))

    return mean_diff, ci_lower, ci_upper, p_val


def holm_bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> List[Tuple[float, bool]]:
    """
    Applies Holm-Bonferroni step-down family-wise error rate correction.
    Returns:
        List of (adjusted_p_value, is_significant) matching original order.
    """
    m = len(p_values)
    if m == 0:
        return []

    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0

    for rank, (orig_idx, p) in enumerate(indexed):
        adj_p = (m - rank) * p
        running_max = max(running_max, adj_p)
        clamped = min(1.0, max(0.0, running_max))
        adjusted[orig_idx] = clamped

    results = []
    for p_adj in adjusted:
        results.append((float(p_adj), bool(p_adj < alpha)))
    return results


def read_results_csv(csv_path: str) -> List[Dict[str, Union[str, float, int]]]:
    """Reads and parses experiments/results.csv into typed record list."""
    records = []
    if not os.path.exists(csv_path):
        return records

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                records.append({
                    "run_id": row["run_id"].strip(),
                    "git_sha": row["git_sha"].strip(),
                    "front_end": row["front_end"].strip(),
                    "family": row["family"].strip(),
                    "seed": int(row["seed"]),
                    "split": row["split"].strip(),
                    "metric": row["metric"].strip(),
                    "value": float(row["value"]),
                })
            except (KeyError, ValueError):
                continue
    return records


def aggregate_by_front_end(
    records: List[Dict[str, Union[str, float, int]]],
    family: str,
    split: str,
    metric: str,
) -> Dict[str, List[float]]:
    """Groups metric values by front_end for a given family, split, and metric."""
    grouped = defaultdict(list)
    for r in records:
        if r["family"] == family and r["split"] == split and r["metric"] == metric:
            grouped[str(r["front_end"])].append(float(r["value"]))
    return dict(grouped)


def generate_summary_table(
    records: List[Dict[str, Union[str, float, int]]],
    family: str,
    split: str = "test",
    metric: str = "macro_auroc",
    candidate_front_ends: Optional[List[str]] = None,
) -> str:
    """Generates markdown summary table with mean ± std, delta vs raw, and Holm p-values."""
    if candidate_front_ends is None:
        candidate_front_ends = ["nlm", "bm3d", "unet_synth", "unet_n2n"]

    grouped = aggregate_by_front_end(records, family=family, split=split, metric=metric)
    raw_vals = np.array(grouped.get("raw", []))

    lines = [
        f"### Family {family.upper()} — {metric} on {split} split",
        "",
        "| Front-End | Mean ± Std (Seeds) | Δ vs Raw (95% CI) | Holm p-value | Supported |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]

    if len(raw_vals) > 0:
        lines.append(f"| `raw` (Baseline) | {raw_vals.mean():.4f} ± {raw_vals.std():.4f} | — | — | Baseline |")
    else:
        lines.append("| `raw` (Baseline) | Missing | — | — | — |")

    # Collect comparisons for Holm correction
    comparisons = []
    p_vals_to_correct = []

    for fe in candidate_front_ends:
        fe_vals = np.array(grouped.get(fe, []))
        if len(fe_vals) > 0 and len(raw_vals) > 0 and len(fe_vals) == len(raw_vals):
            m_diff, ci_l, ci_u, p_raw = compute_paired_bootstrap_ci(raw_vals, fe_vals)
            comparisons.append({
                "front_end": fe,
                "mean": float(fe_vals.mean()),
                "std": float(fe_vals.std()),
                "diff": m_diff,
                "ci": (ci_l, ci_u),
                "p_raw": p_raw,
            })
            p_vals_to_correct.append(p_raw)
        elif len(fe_vals) > 0:
            comparisons.append({
                "front_end": fe,
                "mean": float(fe_vals.mean()),
                "std": float(fe_vals.std()),
                "diff": float("nan"),
                "ci": (float("nan"), float("nan")),
                "p_raw": 1.0,
            })
            p_vals_to_correct.append(1.0)
        else:
            comparisons.append({
                "front_end": fe,
                "mean": float("nan"),
                "std": float("nan"),
                "diff": float("nan"),
                "ci": (float("nan"), float("nan")),
                "p_raw": 1.0,
            })
            p_vals_to_correct.append(1.0)

    holm_results = holm_bonferroni_correction(p_vals_to_correct)

    for item, (p_adj, sig) in zip(comparisons, holm_results):
        fe = item["front_end"]
        if np.isnan(item["mean"]):
            lines.append(f"| `{fe}` | Missing | — | — | — |")
            continue

        mean_str = f"{item['mean']:.4f} ± {item['std']:.4f}"
        if not np.isnan(item["diff"]):
            ci_l, ci_u = item["ci"]
            diff_str = f"{item['diff']:+.4f} [{ci_l:+.4f}, {ci_u:+.4f}]"
            p_str = f"{p_adj:.4f}"
            supp_str = "✅ Yes" if (sig and ci_l > 0) else "❌ No"
        else:
            diff_str = "—"
            p_str = "—"
            supp_str = "—"

        lines.append(f"| `{fe}` | {mean_str} | {diff_str} | {p_str} | {supp_str} |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze results from EXPERIMENT_PLAN.md")
    parser.add_argument("--results", type=str, default="experiments/results.csv")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--metric", type=str, default="macro_auroc")
    args = parser.parse_args()

    records = read_results_csv(args.results)
    print(f"[*] Read {len(records)} records from {args.results}\n")

    for fam in ["a", "b"]:
        tbl = generate_summary_table(records, family=fam, split=args.split, metric=args.metric)
        print(tbl)
        print()


if __name__ == "__main__":
    main()
