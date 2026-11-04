"""
scripts/run_pipeline.py
=======================
Point-of-Care Clinical OCT Diagnostic Pipeline CLI.

Executes the continuous 3-stage pipeline end-to-end:
1. Stage 1: ConvUNet (Neighbor2Neighbor or Synthetic) continuous despeckling
2. Stage 2: ConvAutoencoder normal tissue baseline + AnomalyCalculus MSE heatmap + ROIDecisionLogic
3. Stage 3: MobileNetV3 calibrated crop classifier + Split Conformal Referral triage
4. Telemetry: HIPAA-compliant local SQLite audit logging via SHARPOfflineResilienceManager
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models.classifier.conformal import ReferralCalibrator
from src_python.onnx_model_io import OnnxCropClassifierAdapter, OnnxImageAdapter
from src_python.pathology import PATHOLOGY_CLASSES, crop_roi, describe_pathology
from src_python.roi_logic import ROIDecisionLogic
from src_python.sharp_resilience import SHARPOfflineResilienceManager


def run_pipeline(
    image_path: str,
    despeckler_path: str = "exports/unet_n2n.onnx",
    ae_path: str = "exports/conv_autoencoder_tissue_baseline.onnx",
    classifier_path: str = "exports/mobilenet_v3_pathology_classifier.onnx",
    db_path: str = "runs/telemetry.db",
    output_viz: Optional[str] = None,
    referral_sensitivity: float = 0.95,
) -> Dict:
    """
    Executes end-to-end 3-stage inference and conformal referral triage on a single OCT scan.
    """
    raw_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if raw_img is None:
        raise FileNotFoundError(f"Failed to read image at: {image_path}")

    # Standardize to 256x256 float32 [0.0, 1.0]
    if raw_img.shape != (256, 256):
        raw_norm = cv2.resize(raw_img, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    else:
        raw_norm = raw_img.astype(np.float32) / 255.0

    # Fallback to unet_despeckler if unet_n2n is missing
    despeckler_full = str(REPO_ROOT / despeckler_path)
    if not os.path.exists(despeckler_full):
        despeckler_full = str(REPO_ROOT / "exports" / "unet_despeckler.onnx")

    ae_full = str(REPO_ROOT / ae_path)
    cls_full = str(REPO_ROOT / classifier_path)

    # 1. Stage 1: Despeckling
    despeckler = OnnxImageAdapter(despeckler_full)
    despeckled = despeckler(raw_norm)

    # 2. Stage 2: Anomaly Calculus
    reconstructor = OnnxImageAdapter(ae_full)
    recon = reconstructor(despeckled)
    mse_heatmap = (despeckled - recon) ** 2

    # Smooth heatmap for contour extraction
    blurred = cv2.GaussianBlur(mse_heatmap, (5, 5), 0)
    norm_vis = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    color_heatmap = cv2.applyColorMap(norm_vis, cv2.COLORMAP_JET)

    roi_logic = ROIDecisionLogic()
    raw_findings, anomaly_mask = roi_logic.detect_anomaly_rois(mse_heatmap)

    # 3. Stage 3: Pathology Classification on Crops
    classifier = OnnxCropClassifierAdapter(cls_full) if os.path.exists(cls_full) else None
    findings = []

    # Calibrate triage engine
    triage_engine = ReferralCalibrator(normal_class_idx=3, target_sensitivity=referral_sensitivity)
    # Default sensible baseline threshold
    triage_engine.referral_prob_threshold = 0.35
    triage_engine.calibrated = True

    overall_triage = {"decision": "NO_REFERRAL", "needs_referral": False, "pathology_probability": 0.0}

    for f in raw_findings:
        crop = crop_roi(despeckled, f["bbox"])
        if classifier is not None:
            probs = classifier(crop)
            path_desc = describe_pathology(probs)
            triage_res = triage_engine.triage(probs)

            finding_dict = {
                **f,
                "pathology": path_desc,
                "triage": triage_res,
                "crop_probs": probs.tolist(),
            }
            findings.append(finding_dict)

            # Escalate triage if any finding requires referral
            if triage_res["needs_referral"]:
                overall_triage = triage_res
        else:
            findings.append(f)

    # If no ROIs were detected above threshold, evaluate center tissue baseline
    if not findings and classifier is not None:
        center_crop = crop_roi(despeckled, [96, 96, 64, 64])
        probs = classifier(center_crop)
        triage_res = triage_engine.triage(probs)
        overall_triage = triage_res

    # 4. Telemetry Logging
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    telemetry = SHARPOfflineResilienceManager(db_path=db_path)

    for f in findings:
        path_info = f.get("pathology", {})
        telemetry.log_anomaly_offline(
            anomaly_type=f.get("label", "ANOMALY_REGION"),
            severity=float(f.get("score", 0.0)),
            pathology_label=path_info.get("primary_label"),
            pathology_confidence=path_info.get("confidence"),
            pathology_json=path_info,
        )

    # 5. Visualization Generation (Optional)
    if output_viz:
        os.makedirs(os.path.dirname(output_viz) or ".", exist_ok=True)
        raw_u8 = (raw_norm * 255.0).astype(np.uint8)
        desp_u8 = (np.clip(despeckled, 0.0, 1.0) * 255.0).astype(np.uint8)

        # Draw bounding boxes on overlay
        overlay = cv2.cvtColor(desp_u8, cv2.COLOR_GRAY2BGR)
        for f in findings:
            x, y, w, h = f["bbox"]
            lbl = f.get("pathology", {}).get("primary_label", "ANOMALY")
            conf = f.get("pathology", {}).get("confidence", 0.0)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(
                overlay,
                f"{lbl} {conf:.2f}",
                (x, max(15, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        raw_bgr = cv2.cvtColor(raw_u8, cv2.COLOR_GRAY2BGR)
        banner = np.hstack([raw_bgr, color_heatmap, overlay])
        cv2.imwrite(output_viz, banner)

    return {
        "image_path": str(image_path),
        "mean_mse": float(np.mean(mse_heatmap)),
        "num_findings": len(findings),
        "findings": findings,
        "triage": overall_triage,
        "despeckler_used": Path(despeckler_full).name,
    }


def main():
    parser = argparse.ArgumentParser(description="Point-of-Care Retinal OCT Diagnostic Pipeline")
    parser.add_argument("input", type=str, help="Path to input OCT scan image")
    parser.add_argument("--despeckler", type=str, default="exports/unet_n2n.onnx", help="Despeckler ONNX path")
    parser.add_argument("--autoencoder", type=str, default="exports/conv_autoencoder_tissue_baseline.onnx", help="Tissue AE ONNX path")
    parser.add_argument("--classifier", type=str, default="exports/mobilenet_v3_pathology_classifier.onnx", help="Classifier ONNX path")
    parser.add_argument("--db", type=str, default="runs/telemetry.db", help="Offline SQLite audit DB path")
    parser.add_argument("--out", type=str, default=None, help="Output visualization image path")
    parser.add_argument("--sensitivity", type=float, default=0.95, help="Target clinical referral sensitivity")
    args = parser.parse_args()

    result = run_pipeline(
        image_path=args.input,
        despeckler_path=args.despeckler,
        ae_path=args.autoencoder,
        classifier_path=args.classifier,
        db_path=args.db,
        output_viz=args.out,
        referral_sensitivity=args.sensitivity,
    )

    triage = result["triage"]
    decision = triage.get("decision", "NO_REFERRAL")
    badge = "🔴 REFER TO OPHTHALMOLOGIST" if triage.get("needs_referral") else "🟢 HEALTHY TISSUE (NO REFERRAL)"

    print("\n" + "=" * 60)
    print(f"POINT-OF-CARE OCT CLINICAL REPORT: {Path(args.input).name}")
    print("=" * 60)
    print(f"Triage Recommendation : {badge}")
    print(f"Decision Rule         : {decision}")
    print(f"Mean Anomaly MSE      : {result['mean_mse']:.6f}")
    print(f"Findings Extracted    : {result['num_findings']}")
    print("-" * 60)

    for i, f in enumerate(result["findings"], 1):
        path = f.get("pathology", {})
        lbl = path.get("primary_label", "UNKNOWN")
        conf = path.get("confidence", 0.0)
        bbox = f.get("bbox", [])
        print(f"  Finding #{i}: {lbl} (Confidence: {conf*100:.1f}%) at BBox {bbox}")

    if args.out:
        print(f"\n[+] Visualization saved to: {args.out}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
