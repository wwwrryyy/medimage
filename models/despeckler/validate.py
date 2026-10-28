"""
validate.py
===========
PyTorch ↔ ONNX Runtime equivalence validation and latency benchmarking for ConvUNet.

Asserts that ONNX outputs match native PyTorch within 10⁻⁴ tolerance and measures
mean edge inference latency against the ≥15 FPS (≤66.6 ms) target.
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src_python"))
from onnx_model_io import OnnxImageAdapter, verify_pytorch_onnx_parity

try:
    from model import ConvUNet
except ImportError:
    from models.despeckler.model import ConvUNet


def validate(args):
    device = torch.device("cpu")

    # --- Load PyTorch model ---
    model = ConvUNet(
        in_channels=1,
        base_ch=args.base_ch,
    ).to(device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"[+] Loaded PyTorch checkpoint: {args.checkpoint}")

    model.eval()

    # --- Load ONNX model ---
    if not os.path.exists(args.onnx):
        print(f"[!] Error: ONNX file not found at {args.onnx}")
        return 1

    adapter = OnnxImageAdapter(args.onnx)
    print(f"[+] Loaded ONNX model: {args.onnx}")

    # --- Run Numerical Parity Comparison ---
    print(f"[*] Testing PyTorch vs ONNX Runtime numerical equivalence on {args.num_samples} samples...")
    all_passed, max_diff_seen, num_passed = verify_pytorch_onnx_parity(
        pytorch_model=model,
        onnx_path=args.onnx,
        height=args.height,
        width=args.width,
        in_channels=1,
        num_samples=args.num_samples,
        tolerance=args.tolerance,
        seed=args.seed,
    )

    print(f"\n[*] Numerical Parity: {num_passed}/{args.num_samples} passed (tolerance={args.tolerance:.0e})")
    print(f"    Max absolute diff: {max_diff_seen:.6e}")

    # --- Latency Benchmarking ---
    print(f"[*] Running latency benchmark ({args.benchmark_iters} iterations on CPU)...")
    stats = adapter.benchmark(height=args.height, width=args.width, iters=args.benchmark_iters)
    mean_lat = stats["mean_ms"]
    p95_lat = stats["p95_ms"]
    fps = stats["fps"]

    print(f"[+] ConvUNet ONNX Runtime CPU Latency ({args.height}x{args.width}):")
    print(f"    - Mean: {mean_lat:.2f} ms")
    print(f"    - P95:  {p95_lat:.2f} ms")
    print(f"    - Throughput: {fps:.1f} FPS (Target: ≥15.0 FPS)")

    if fps >= 15.0:
        print(f"    - Status: [PASS] Exceeds 15 FPS edge budget.")
    else:
        print(f"    - Status: [WARN] Below 15 FPS on this host.")

    if all_passed:
        print(f"[+] VALIDATION PASSED for ConvUNet!")
        return 0
    else:
        print(f"[!] VALIDATION FAILED: Numerical divergence.")
        return 1


def parse_args():
    p = argparse.ArgumentParser(description="Validate PyTorch ↔ ONNX equivalence and benchmark latency")
    p.add_argument("--checkpoint", type=str, default="exports/despeckler_cnn_best.pt")
    p.add_argument("--onnx", type=str, default="exports/unet_despeckler.onnx")
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--tolerance", type=float, default=1e-4)
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--benchmark-iters", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(validate(parse_args()))
