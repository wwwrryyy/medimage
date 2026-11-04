"""
tests/test_pipeline_cli.py
==========================
Unit tests for point-of-care clinical diagnostic pipeline CLI.
"""

import os
from pathlib import Path
import pytest

from scripts.run_pipeline import run_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_run_pipeline_end_to_end(tmp_path):
    sample_scan = str(REPO_ROOT / "calibration_data" / "pathology" / "anon_path_DME_016.png")
    if not os.path.exists(sample_scan):
        pytest.skip("Sample scan not found.")

    out_viz = str(tmp_path / "out_viz.png")
    db_path = str(tmp_path / "test_telemetry.db")

    result = run_pipeline(
        image_path=sample_scan,
        db_path=db_path,
        output_viz=out_viz,
        referral_sensitivity=0.95,
    )

    assert "mean_mse" in result
    assert result["mean_mse"] >= 0.0
    assert "triage" in result
    assert "decision" in result["triage"]
    assert "needs_referral" in result["triage"]
    assert os.path.exists(out_viz)
    assert os.path.exists(db_path)
