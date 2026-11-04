"""
tests/test_n2n.py
=================
Tests for Neighbor2Neighbor (N2N) self-supervised despeckler pipeline.
"""

import os
import torch
import numpy as np
import pytest

from models.despeckler.train_n2n import generate_subpixel_neighbor_pair, train_n2n
from models.despeckler.model import ConvUNet


def test_subpixel_neighbor_sampling_shapes():
    x = torch.randn(2, 1, 64, 64)
    sub1, sub2 = generate_subpixel_neighbor_pair(x)
    assert sub1.shape == (2, 1, 32, 32)
    assert sub2.shape == (2, 1, 32, 32)
    # Check distinct sampling points
    assert not torch.equal(sub1, sub2)


def test_train_n2n_smoke(tmp_path):
    output_dir = str(tmp_path / "exports")
    ckpt = train_n2n(
        output_dir=output_dir,
        epochs=1,
        batch_size=2,
        lr=1e-3,
        seed=42,
        max_samples=4,
        force_cpu=True,
    )
    assert os.path.exists(ckpt)
    onnx_file = os.path.join(output_dir, "unet_n2n.onnx")
    assert os.path.exists(onnx_file)
