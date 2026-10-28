"""
onnx_model_io.py
================
Shared ONNX Runtime inference, benchmarking, PyTorch -> ONNX export, and
numerical equivalence verification across Stage 1 (ConvUNet) and Stage 2 (ConvAutoencoder).
"""

import os
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class OnnxImageAdapter:
    """
    ONNX Runtime adapter for 2D float32 [H, W] in [0.0, 1.0] -> 2D float32 [H, W] in [0.0, 1.0].

    Automatically adapts arbitrary frame dimensions to fixed ONNX spatial dimensions
    (e.g. 256x256 in unet_despeckler.onnx) or pads dynamic dimensions to multiples of 4
    for 2-stride encoder/decoder architectures, then restores the original resolution.
    """

    def __init__(self, onnx_path: str, providers: Optional[List[str]] = None):
        import onnxruntime as ort

        self.onnx_path = onnx_path
        self.session = ort.InferenceSession(
            onnx_path,
            providers=providers or ["CPUExecutionProvider"],
        )
        inp = self.session.get_inputs()[0]
        self.input_name: str = inp.name
        self.output_name: str = self.session.get_outputs()[0].name

        shape = inp.shape
        self.fixed_h: Optional[int] = shape[2] if len(shape) >= 4 and isinstance(shape[2], int) else None
        self.fixed_w: Optional[int] = shape[3] if len(shape) >= 4 and isinstance(shape[3], int) else None

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 2:
            raise ValueError(f"Expected 2D float32 image [H, W], got shape {image.shape}")

        orig_h, orig_w = image.shape
        work = image.astype(np.float32, copy=False)

        target_h = self.fixed_h or ((orig_h + 3) // 4) * 4
        target_w = self.fixed_w or ((orig_w + 3) // 4) * 4

        if (target_h, target_w) != (orig_h, orig_w):
            interp = cv2.INTER_AREA if (target_h < orig_h or target_w < orig_w) else cv2.INTER_LINEAR
            work = cv2.resize(work, (target_w, target_h), interpolation=interp)

        tensor = work[np.newaxis, np.newaxis, :, :]
        pred = self.session.run([self.output_name], {self.input_name: tensor})[0][0, 0]
        pred = np.clip(pred, 0.0, 1.0).astype(np.float32)

        if (target_h, target_w) != (orig_h, orig_w):
            pred = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        return pred

    def benchmark(self, height: int = 256, width: int = 256, iters: int = 50, warmup: int = 5) -> Dict[str, float]:
        """Measures CPU latency (mean_ms, p95_ms) and throughput (fps) on random input."""
        return benchmark_onnx_session(self.session, height=height, width=width, iters=iters, warmup=warmup)


class OnnxCropClassifierAdapter:
    """
    ONNX Runtime adapter for the Stage 3 classifier: 2D float32 crop [H, W] in [0.0, 1.0]
    -> 1D float32 class-probability vector [C]. Crops are resized to the model's fixed
    input resolution if they differ.
    """

    def __init__(self, onnx_path: str, providers: Optional[List[str]] = None):
        import onnxruntime as ort

        self.onnx_path = onnx_path
        self.session = ort.InferenceSession(
            onnx_path,
            providers=providers or ["CPUExecutionProvider"],
        )
        inp = self.session.get_inputs()[0]
        self.input_name: str = inp.name
        self.output_name: str = self.session.get_outputs()[0].name

        shape = inp.shape
        self.fixed_h: Optional[int] = shape[2] if len(shape) >= 4 and isinstance(shape[2], int) else None
        self.fixed_w: Optional[int] = shape[3] if len(shape) >= 4 and isinstance(shape[3], int) else None

    def __call__(self, crop: np.ndarray) -> np.ndarray:
        if crop.ndim != 2:
            raise ValueError(f"Expected 2D float32 crop [H, W], got shape {crop.shape}")

        work = crop.astype(np.float32, copy=False)
        if self.fixed_h and self.fixed_w and work.shape != (self.fixed_h, self.fixed_w):
            work = cv2.resize(work, (self.fixed_w, self.fixed_h), interpolation=cv2.INTER_AREA)

        tensor = work[np.newaxis, np.newaxis, :, :]
        probs = self.session.run([self.output_name], {self.input_name: tensor})[0][0]
        return probs.astype(np.float32)


def benchmark_onnx_session(
    session,
    height: int = 256,
    width: int = 256,
    iters: int = 50,
    warmup: int = 5,
    seed: int = 42,
) -> Dict[str, float]:
    """Benchmarks an ONNX Runtime InferenceSession and returns mean_ms, p95_ms, and fps."""
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    rng = np.random.default_rng(seed)
    dummy = rng.random((1, 1, height, width)).astype(np.float32)

    for _ in range(warmup):
        session.run([output_name], {input_name: dummy})

    latencies: List[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        session.run([output_name], {input_name: dummy})
        latencies.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = float(np.mean(latencies))
    p95_ms = float(np.percentile(latencies, 95))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    return {"mean_ms": mean_ms, "p95_ms": p95_ms, "fps": fps}


def export_pytorch_to_onnx(
    model,
    output_path: str,
    height: int = 256,
    width: int = 256,
    in_channels: int = 1,
    input_name: str = "input_frame",
    output_name: str = "output_frame",
    dynamic_spatial: bool = True,
    opset_version: int = 17,
) -> str:
    """Exports a 2D image-to-image PyTorch model to ONNX with dynamic batch (and optional spatial) axes."""
    import torch

    cpu_model = model.to("cpu").eval()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    dummy = torch.randn(1, in_channels, height, width, device="cpu")

    if dynamic_spatial:
        dynamic_axes = {
            input_name: {0: "batch_size", 2: "height", 3: "width"},
            output_name: {0: "batch_size", 2: "height", 3: "width"},
        }
    else:
        dynamic_axes = {
            input_name: {0: "batch_size"},
            output_name: {0: "batch_size"},
        }

    torch.onnx.export(
        cpu_model,
        dummy,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=[input_name],
        output_names=[output_name],
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )
    return output_path


def verify_pytorch_onnx_parity(
    pytorch_model,
    onnx_path: str,
    height: int = 256,
    width: int = 256,
    in_channels: int = 1,
    num_samples: int = 10,
    tolerance: float = 1e-4,
    seed: int = 42,
) -> Tuple[bool, float, int]:
    """
    Verifies numerical equivalence between a PyTorch model and an exported ONNX graph.
    Returns (all_passed, max_abs_diff, num_passed).
    """
    import onnxruntime as ort
    import torch

    cpu_model = pytorch_model.to("cpu").eval()
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    rng = np.random.default_rng(seed)
    num_passed = 0
    max_diff_seen = 0.0

    for _ in range(num_samples):
        x_np = rng.random((1, in_channels, height, width)).astype(np.float32)
        with torch.no_grad():
            y_pt = cpu_model(torch.from_numpy(x_np)).numpy()
        y_ort = session.run([output_name], {input_name: x_np})[0]

        diff = float(np.max(np.abs(y_pt - y_ort)))
        max_diff_seen = max(max_diff_seen, diff)
        if diff <= tolerance:
            num_passed += 1

    return (num_passed == num_samples), max_diff_seen, num_passed
