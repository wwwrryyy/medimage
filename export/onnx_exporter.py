"""
onnx_exporter.py
================
Unified ONNX Export and Numerical Parity Verification CLI for all 3 pipeline stages:
1. Stage 1: ConvUNet Despeckler (models/despeckler/model.py)
2. Stage 2: ConvAutoencoder Tissue Baseline (models/anomaly_detector/autoencoder.py)
3. Stage 3: MobileNetV3 Pathology Classifier (models/classifier/mobilenet_v3.py)

Outputs all models to exports/*.onnx with verified tolerance <= 1e-4.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src_python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from onnx_model_io import export_pytorch_to_onnx, verify_pytorch_onnx_parity

from models.despeckler.model import ConvUNet
from models.anomaly_detector.autoencoder import ConvAutoencoder
from models.classifier.mobilenet_v3 import MobileNetV3Classifier, CalibratedClassifier


def export_stage1_despeckler(
    checkpoint: str,
    output_path: str,
    base_ch: int = 32,
    height: int = 256,
    width: int = 256,
    tolerance: float = 1e-4,
) -> Dict[str, Any]:
    print("\n" + "=" * 60)
    print("STAGE 1: ConvUNet Despeckler Export")
    print("=" * 60)
    model = ConvUNet(in_channels=1, base_ch=base_ch).eval()
    if checkpoint and os.path.exists(checkpoint):
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        print(f"[+] Loaded weights: {checkpoint}")
    else:
        print("[!] Warning: Checkpoint not found, exporting default/initialized architecture.")

    export_pytorch_to_onnx(
        model=model,
        output_path=output_path,
        height=height,
        width=width,
        in_channels=1,
        input_name="corrupted_frame",
        output_name="despeckled_frame",
        dynamic_spatial=True,
    )
    ok, max_diff, num_passed = verify_pytorch_onnx_parity(
        pytorch_model=model,
        onnx_path=output_path,
        height=height,
        width=width,
        in_channels=1,
        tolerance=tolerance,
    )
    size_kb = os.path.getsize(output_path) / 1024
    print(f"[+] Exported: {output_path} ({size_kb:.1f} KB)")
    print(f"[*] Parity: {'PASSED' if ok else 'FAILED'} (max_diff={max_diff:.2e}, tolerance={tolerance:.0e})")
    return {"stage": 1, "model": "ConvUNet", "path": output_path, "size_kb": size_kb, "parity_ok": ok, "max_diff": max_diff}


def export_stage2_autoencoder(
    checkpoint: str,
    output_path: str,
    height: int = 128,
    width: int = 128,
    tolerance: float = 1e-4,
) -> Dict[str, Any]:
    print("\n" + "=" * 60)
    print("STAGE 2: ConvAutoencoder Tissue Baseline Export")
    print("=" * 60)
    model = ConvAutoencoder(in_channels=1).eval()
    if checkpoint and os.path.exists(checkpoint):
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        print(f"[+] Loaded weights: {checkpoint}")
    else:
        print("[!] Warning: Checkpoint not found, exporting default/initialized architecture.")

    export_pytorch_to_onnx(
        model=model,
        output_path=output_path,
        height=height,
        width=width,
        in_channels=1,
        input_name="input_frame",
        output_name="reconstructed_baseline",
        dynamic_spatial=True,
    )
    ok, max_diff, num_passed = verify_pytorch_onnx_parity(
        pytorch_model=model,
        onnx_path=output_path,
        height=height,
        width=width,
        in_channels=1,
        tolerance=tolerance,
    )
    size_kb = os.path.getsize(output_path) / 1024
    print(f"[+] Exported: {output_path} ({size_kb:.1f} KB)")
    print(f"[*] Parity: {'PASSED' if ok else 'FAILED'} (max_diff={max_diff:.2e}, tolerance={tolerance:.0e})")
    return {"stage": 2, "model": "ConvAutoencoder", "path": output_path, "size_kb": size_kb, "parity_ok": ok, "max_diff": max_diff}


def export_stage3_classifier(
    checkpoint: str,
    output_path: str,
    crop_size: int = 64,
    tolerance: float = 1e-4,
) -> Dict[str, Any]:
    print("\n" + "=" * 60)
    print("STAGE 3: MobileNetV3 Pathology Classifier Export")
    print("=" * 60)
    model = MobileNetV3Classifier(num_classes=4, in_channels=1).eval()
    if checkpoint and os.path.exists(checkpoint):
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        print(f"[+] Loaded weights: {checkpoint}")
    else:
        print("[!] Warning: Checkpoint not found, exporting default/initialized architecture.")

    wrapped = CalibratedClassifier(model)
    export_pytorch_to_onnx(
        model=wrapped,
        output_path=output_path,
        height=crop_size,
        width=crop_size,
        in_channels=1,
        input_name="roi_crop",
        output_name="class_probabilities",
        dynamic_spatial=False,
    )
    ok, max_diff, num_passed = verify_pytorch_onnx_parity(
        pytorch_model=wrapped,
        onnx_path=output_path,
        height=crop_size,
        width=crop_size,
        in_channels=1,
        tolerance=tolerance,
    )
    size_kb = os.path.getsize(output_path) / 1024
    print(f"[+] Exported: {output_path} ({size_kb:.1f} KB)")
    print(f"[*] Parity: {'PASSED' if ok else 'FAILED'} (max_diff={max_diff:.2e}, tolerance={tolerance:.0e})")
    return {"stage": 3, "model": "MobileNetV3", "path": output_path, "size_kb": size_kb, "parity_ok": ok, "max_diff": max_diff}


def main():
    parser = argparse.ArgumentParser(description="Unified ONNX Exporter for Stages 1, 2, and 3")
    parser.add_argument("--stage", type=str, choices=["all", "1", "2", "3"], default="all", help="Stage to export")
    parser.add_argument("--exports-dir", type=str, default="exports", help="Directory for exported ONNX files")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Max absolute difference tolerance")
    args = parser.parse_args()

    exports_dir = Path(args.exports_dir)
    exports_dir.mkdir(parents=True, exist_ok=True)

    results = []

    if args.stage in ("all", "1"):
        res1 = export_stage1_despeckler(
            checkpoint=str(exports_dir / "despeckler_cnn_best.pt"),
            output_path=str(exports_dir / "unet_despeckler.onnx"),
            tolerance=args.tolerance,
        )
        results.append(res1)

    if args.stage in ("all", "2"):
        res2 = export_stage2_autoencoder(
            checkpoint=str(exports_dir / "conv_autoencoder_tissue_baseline.pt"),
            output_path=str(exports_dir / "conv_autoencoder_tissue_baseline.onnx"),
            tolerance=args.tolerance,
        )
        results.append(res2)

    if args.stage in ("all", "3"):
        stage3_path = str(exports_dir / "mobilenet_v3_pathology_classifier.onnx")
        res3 = export_stage3_classifier(
            checkpoint=str(exports_dir / "pathology_classifier_best.pt"),
            output_path=stage3_path,
            tolerance=args.tolerance,
        )
        results.append(res3)
        # Maintain backward-compatible copy for any legacy paths
        legacy_path = str(exports_dir / "pathology_classifier.onnx")
        if os.path.exists(stage3_path) and stage3_path != legacy_path:
            import shutil
            shutil.copyfile(stage3_path, legacy_path)

    print("\n" + "=" * 70)
    print("UNIFIED ONNX EXPORT SUMMARY")
    print("=" * 70)
    print(f"| Stage | Model              | Size (KB) | Parity Status | Max Diff  |")
    print(f"|-------|--------------------|-----------|---------------|-----------|")
    all_ok = True
    for r in results:
        status = "PASSED" if r["parity_ok"] else "FAILED"
        if not r["parity_ok"]:
            all_ok = False
        print(f"| {r['stage']:<5} | {r['model']:<18} | {r['size_kb']:>9.1f} | {status:<13} | {r['max_diff']:>9.2e} |")
    print("=" * 70)

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
