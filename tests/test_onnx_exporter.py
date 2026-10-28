"""Unit tests for Unified ONNX Exporter."""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from export.onnx_exporter import (
    export_stage1_despeckler,
    export_stage2_autoencoder,
    export_stage3_classifier,
)


def test_unified_export_stage1(tmp_path):
    out_path = str(tmp_path / "unet.onnx")
    res = export_stage1_despeckler(
        checkpoint="exports/despeckler_cnn_best.pt",
        output_path=out_path,
        height=64,
        width=64,
    )
    assert res["parity_ok"] is True
    assert res["max_diff"] <= 1e-4
    assert os.path.exists(out_path)


def test_unified_export_stage2(tmp_path):
    out_path = str(tmp_path / "autoencoder.onnx")
    res = export_stage2_autoencoder(
        checkpoint="exports/conv_autoencoder_tissue_baseline.pt",
        output_path=out_path,
        height=64,
        width=64,
    )
    assert res["parity_ok"] is True
    assert res["max_diff"] <= 1e-4
    assert os.path.exists(out_path)


def test_unified_export_stage3(tmp_path):
    out_path = str(tmp_path / "classifier.onnx")
    res = export_stage3_classifier(
        checkpoint="exports/pathology_classifier_best.pt",
        output_path=out_path,
        crop_size=64,
    )
    assert res["parity_ok"] is True
    assert res["max_diff"] <= 1e-4
    assert os.path.exists(out_path)
