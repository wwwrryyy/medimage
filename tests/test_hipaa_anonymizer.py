"""Unit tests for HIPAA Safe Harbor Anonymization and metadata stripping."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src_python"))
from hipaa_anonymizer import HIPAAAnonymizer


def test_phi_detection_flags_patient_identifiers():
    assert len(HIPAAAnonymizer.check_string_for_phi("NORMAL-100980-1.jpeg")) == 0
    assert len(HIPAAAnonymizer.check_string_for_phi("patient_john_doe_scan.png")) > 0
    assert len(HIPAAAnonymizer.check_string_for_phi("scan_mrn_1234567.jpg")) > 0
    assert len(HIPAAAnonymizer.check_string_for_phi("oct_123-45-6789.png")) > 0


def test_strip_metadata_produces_clean_image(tmp_path):
    # Create image with EXIF metadata
    src_path = tmp_path / "raw_with_exif.jpg"
    img = Image.fromarray(np.full((128, 128), 128, dtype=np.uint8))
    exif = img.getexif()
    exif[0x010E] = "Patient: John Doe"  # ImageDescription tag
    img.save(src_path, exif=exif)

    # Verify input has EXIF
    with Image.open(src_path) as im:
        assert len(im.getexif()) > 0

    # Anonymize
    out_path = tmp_path / "anon_clean.png"
    result = HIPAAAnonymizer.strip_metadata_and_save(src_path, out_path)
    assert result["success"] is True

    # Verify output has zero EXIF
    assert HIPAAAnonymizer.verify_clean(out_path) is True
    with Image.open(out_path) as im:
        assert len(im.getexif()) == 0


def test_banner_masking_blanks_margins(tmp_path):
    src_path = tmp_path / "scan_banner.png"
    arr = np.full((100, 100), 200, dtype=np.uint8)
    Image.fromarray(arr).save(src_path)

    out_path = tmp_path / "masked.png"
    HIPAAAnonymizer.strip_metadata_and_save(
        src_path, out_path, top_margin_mask_px=20, bottom_margin_mask_px=15
    )

    masked_arr = np.array(Image.open(out_path))
    assert np.all(masked_arr[:20, :] == 0)      # Top banner blanked
    assert np.all(masked_arr[85:, :] == 0)      # Bottom banner blanked
    assert np.all(masked_arr[25:80, :] == 200)  # Core scan untouched
