"""Unit tests for INT8 calibration data and data reader."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from calibration_data.calibration_reader import Int8CalibrationDataReader


def test_calibration_data_directory_exists_and_has_samples():
    norm_dir = Path("calibration_data/normal")
    path_dir = Path("calibration_data/pathology")
    assert norm_dir.is_dir()
    assert path_dir.is_dir()
    norm_files = list(norm_dir.glob("*.png"))
    path_files = list(path_dir.glob("*.png"))
    assert len(norm_files) >= 20
    assert len(path_files) >= 20


def test_int8_calibration_data_reader():
    reader = Int8CalibrationDataReader(
        data_dir="calibration_data",
        input_name="corrupted_frame",
        height=128,
        width=128,
    )
    assert len(reader) >= 40

    sample = reader.get_next()
    assert sample is not None
    assert "corrupted_frame" in sample
    tensor = sample["corrupted_frame"]
    assert tensor.shape == (1, 1, 128, 128)
    assert tensor.dtype.name == "float32"
    assert 0.0 <= float(tensor.min()) <= float(tensor.max()) <= 1.0

    reader.rewind()
    sample2 = reader.get_next()
    assert sample2 is not None
