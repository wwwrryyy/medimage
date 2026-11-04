# MedImage: Point-of-Care Retinal OCT Diagnostic Pipeline

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-66%20passed-brightgreen.svg)]()
[![Inference Speed](https://img.shields.io/badge/edge%20speed-160.7%20FPS-blueviolet.svg)]()

An edge-deployable, continuous deep learning pipeline for real-time Optical Coherence Tomography (OCT) image enhancement, unsupervised lesion anomaly detection, and calibrated diagnostic triage. Designed for point-of-care and rural clinical environments.

---

## Key Features

- **Continuous 3-Stage Pipeline**:
  1. **Stage 1 (Speckle Filter)**: Single-pass continuous `ConvUNet` trained via self-supervised Neighbor2Neighbor (`unet_n2n`) without synthetic noise bias. Runs at **160.7 FPS (6.22 ms/frame)** on CPU.
  2. **Stage 2 (Anomaly Detection)**: `ConvAutoencoder` trained on healthy tissue baselines paired with a high-performance C++ `AnomalyCalculus` engine and spatial Non-Maximum Suppression (NMS) to localize retinal lesions (fluid pockets, drusen, neovascularization).
  3. **Stage 3 (Pathology Classification & Triage)**: Temperature-scaled `MobileNetV3` crop classifier paired with **Split Conformal Prediction** to provide clinically guaranteed $\ge 95\%$ sensitivity for ophthalmologist referrals.
- **Offline Resilience (SHARP)**: Local encrypted SQLite telemetry buffering with automatic reconnection sync for rural or disconnected field clinics.
- **Privacy First**: Built-in HIPAA/GDPR de-identification utilities to scrub metadata and patient identifiers prior to processing.
- **Hardware Agnostic**: Fully exported ONNX models with verified numerical equivalence ($\le 10^{-4}$) and embedded ARM cross-compilation toolchains.

---

## Pipeline Workflow

```text
[ Raw Retinal OCT Frame ]
           │
           ▼
┌──────────────────────────────────────────────┐
│  Stage 1: ConvUNet Despeckler (unet_n2n)     │ ──▶ Suppresses multiplicative speckle
└──────────────────────────────────────────────┘      without blurring pathology margins
           │
           ▼
┌──────────────────────────────────────────────┐
│  Stage 2: ConvAutoencoder Anomaly Calculus   │ ──▶ Compares against healthy baseline
└──────────────────────────────────────────────┘      to produce pixel-wise MSE heatmaps
           │
           ▼
┌──────────────────────────────────────────────┐
│  ROIDecisionLogic & NMS Box Extraction       │ ──▶ Extracts 64x64 candidate lesion crops
└──────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────┐
│  Stage 3: MobileNetV3 + Conformal Triage     │ ──▶ Classifies CNV, DME, DRUSEN, NORMAL
└──────────────────────────────────────────────┘      with mathematically guaranteed referral
           │
           ▼
[ SHARP Offline SQLite Telemetry Audit Log ]
```

---

## Empirical Benchmark Results

Evaluated across the **Kermany Retinal OCT** dataset (patient-level stratified) and **Duke DME 2015** clinical 3D volumes (Chiu et al., BOE 2015):

| Downstream Family | Front-End Filter | Macro AUROC | Test ECE (15 bins) | Fluid AUPRO@0.3 | Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Family A (MIL Proposals)** | `raw` (Baseline) | $0.7889 \pm 0.0100$ | $0.0445$ | $0.2164$ | $0.0\text{ ms}$ |
| **Family A (MIL Proposals)** | **`unet_n2n` (Ours)** | **$0.7741 \pm 0.0054$** | **$0.0414$** | **$0.2601$ (+20.2%)** | **$6.22\text{ ms}$** |
| **Family A (MIL Proposals)** | `unet_synth` | $0.7450 \pm 0.0099$ | $0.0541$ | $0.1023$ (-52.7%) | $6.22\text{ ms}$ |
| **Family B (Whole Scan)** | `raw` (Baseline) | $0.9749 \pm 0.0014$ | $0.0366$ | $0.6261$ | $0.0\text{ ms}$ |
| **Family B (Whole Scan)** | **`unet_n2n` (Ours)** | **$0.9750 \pm 0.0036$** | **$0.0336$** | $0.4351$ | **$6.22\text{ ms}$** |
| **Family B (Whole Scan)** | `unet_synth` | $0.9807 \pm 0.0049$ | $0.0303$ | — | $6.22\text{ ms}$ |
| **Classical Reference** | `bm3d` | — | — | — | $\sim 1200.0\text{ ms}$ (< 1 FPS) |

---

## Quickstart

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone git@github.com:wwwrryyy/medical-imaging.git
cd medical-imaging

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Run Diagnostic Pipeline on an Image

Analyze any OCT scan with the complete 3-stage pipeline, evaluate conformal triage, and save a multi-panel visual overlay:

```bash
python scripts/run_pipeline.py calibration_data/pathology/anon_path_DME_016.png --out report_overlay.png
```

**Terminal Output:**
```text
============================================================
POINT-OF-CARE OCT CLINICAL REPORT: anon_path_DME_016.png
============================================================
Triage Recommendation : 🔴 REFER TO OPHTHALMOLOGIST
Decision Rule         : REFER (Sensitivity >= 95% Guarantee)
Mean Anomaly MSE      : 0.023644
Findings Extracted    : 1
------------------------------------------------------------
  Finding #1: DME (Confidence: 89.4%) at BBox [102, 114, 64, 64]

[+] Visualization saved to: report_overlay.png
============================================================
```

### 3. Run the Automated Test Suite

Run the full suite of unit and integration tests:

```bash
pytest tests/
# 66 passed, 3 skipped in ~21s
```

### 4. Reproduce Experiments

Train and evaluate any pre-registered arm across multiple seeds:

```bash
python experiments/run.py experiments/configs/family_a_unet_n2n_seed42.yaml
python experiments/analyze.py --split test --metric macro_auroc
```

---

## Repository Structure

```text
medimage/
├── calibration_data/       # INT8 quantization calibration scans & reader
├── data_scripts/           # Patient-level splitting, deduplication, Duke DME unpacker
├── docs/                   # ARM embedded cross-compilation documentation
├── experiments/            # Automated multi-seed runner, configs, results database
├── export/                 # Unified ONNX exporter with numerical parity validation
├── exports/                # Ready-to-deploy continuous ONNX model weights
├── models/
│   ├── anomaly_detector/   # Stage 2: ConvAutoencoder normal tissue baseline
│   ├── classifier/         # Stage 3: MobileNetV3 & Conformal Prediction referral
│   └── despeckler/         # Stage 1: Continuous ConvUNet & Neighbor2Neighbor
├── scripts/                # run_pipeline CLI, confidence & NMS calibration
├── src_cpp/                # C++ AnomalyCalculus engine & pybind11 zero-copy bridge
├── src_python/             # ROI logic, ScanSession, HIPAA anonymizer, SHARP SQLite
└── tests/                  # Automated pytest verification test suite
```

---

## Clinical & Regulatory Disclaimer

> [!IMPORTANT]
> This software is an algorithmic research prototype developed for pre-clinical investigations and computational benchmarking. It is not currently certified as a Medical Device (SaMD) and has not received FDA 510(k) clearance or CE mark approval. It is not intended for primary clinical diagnosis or direct patient management without qualified ophthalmologist oversight.

---

## License

This project is licensed under the Apache 2.0 License — see the [LICENSE](LICENSE) file for details.
