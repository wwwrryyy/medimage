"""
export.py
=========
PyTorch → ONNX export for Stage 1 Despeckler (ConvUNet).

Produces a clean, single-pass continuous computation graph (T=1) compatible with
ONNX Runtime CPU / ARM for embedded point-of-care deployment.
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src_python"))
from onnx_model_io import export_pytorch_to_onnx

try:
    from model import ConvUNet
except ImportError:
    from models.despeckler.model import ConvUNet


def export_onnx(args):
    device = torch.device("cpu")  # Export on CPU for portability

    model = ConvUNet(
        in_channels=1,
        base_ch=args.base_ch,
    ).to(device)

    # Load trained weights if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"[+] Loaded weights checkpoint: {args.checkpoint}")
    elif args.checkpoint:
        print(f"[!] Warning: checkpoint {args.checkpoint} not found. Exporting initialized architecture.")

    output_path = args.output or os.path.join("exports", "unet_despeckler.onnx")
    print(f"[*] Exporting ConvUNet model to ONNX: {output_path}")

    export_pytorch_to_onnx(
        model=model,
        output_path=output_path,
        height=args.height,
        width=args.width,
        in_channels=1,
        input_name="corrupted_frame",
        output_name="clean_frame",
        dynamic_spatial=True,
        opset_version=args.opset,
    )

    size_kb = os.path.getsize(output_path) / 1024
    print(f"[+] Successfully exported ConvUNet to: {output_path} ({size_kb:.1f} KB)")
    return output_path


def parse_args():
    p = argparse.ArgumentParser(description="Export ConvUNet Despeckler to ONNX")
    p.add_argument("--checkpoint", type=str, default="exports/despeckler_cnn_best.pt",
                    help="Path to trained .pt checkpoint")
    p.add_argument("--output", type=str, default="exports/unet_despeckler.onnx",
                    help="Output ONNX file path")
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--opset", type=int, default=17)
    return p.parse_args()


if __name__ == "__main__":
    export_onnx(parse_args())
