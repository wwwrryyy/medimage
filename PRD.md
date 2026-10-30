PRD 1: Medical Imaging & Point-of-Care Diagnostics (Edge-Deployed CNN Pipeline)

1. Objective
⚬ Deploy optimized, continuous lightweight CNN architectures for portable medical image enhancement and anomaly detection.
⚬ Target: handheld battery-powered ARM devices (e.g., portable OCT scanners) at ≥15 FPS. Not yet measured on ARM hardware; current Stage 1 benchmark is 6.2 ms/frame on an Apple M5 Mac CPU (ONNX Runtime).
⚬ Serve as a pre-clinical research tool; note that clinical deployment requires regulatory validation (FDA 510(k) / CE marking).
⚬ Ensure patient data handling is HIPAA/GDPR compliant with strict anonymization requirements.

2. Data Synthesis & Preprocessing
⚬ OCT scans suffer from multiplicative speckle noise characterized by Rayleigh and Gamma intensity distributions.
⚬ Calibrated corruption engine (`models/despeckler/corruption.py`) generates strict 1:1 corrupted→clean training pairs matched to clinical speckle contrast (IQR: 0.116 - 0.140).
⚬ Train the Stage 2 model on strictly healthy/normal tissue scans (negative-only training, ZERO pathological samples).
⚬ Scope is retinal OCT only. Datasets utilized: Duke SD-OCT (Fang 2012), Kermany Retinal OCT (currently a 1,000-image subset; full OCT2017 planned). Lesion-level evaluation planned on RETOUCH / Duke DME 2015 (see EXPERIMENT_PLAN.md).

3. Model Architecture
Component	Specification
Stage 1: OCT Despeckler	ConvUNet (models/despeckler/model.py); Continuous 3-stage encoder-decoder with SiLU activations and skip connections; Single-pass continuous inference (T=1, 6.2 ms on Apple M5 Mac CPU); MSE + SSIM loss; 28.79 dB PSNR / 0.8633 SSIM on clinical Duke SD-OCT (below BM3D: 29.03 dB / 0.8834)
Stage 2: Tissue Anomaly Detector	ConvAutoencoder (models/anomaly_detector/autoencoder.py); Continuous convolutional autoencoder; Learns normal tissue baseline and flags pathological lesions via reconstruction error; Pixel-wise MSE subtraction for anomaly heatmaps
Stage 2: Processing Engine	AnomalyCalculus (src_cpp/include/anomaly_calculus.hpp); C++ pixel-wise MSE computation between despeckled input and autoencoder reconstructed baseline; Gaussian smoothing; Visual heatmap generation; Zero-copy pybind11 bridge
Stage 2: Clustering Pipeline	ROIDecisionLogic (src_python/roi_logic.py); Morphological filtering; Contour extraction with area filtering; Non-Maximum Suppression (NMS) with IoU for ROI bounding boxes
Stage 3: Pathology Classifier	MobileNetV3 Classifier; Depthwise separable convolutions; Temperature scaling for calibrated clinical confidence scores; 64×64 fixed input crops; Cross-Entropy loss
Infrastructure: Zero-Copy Bridge	pybind11 (src_cpp/src/pybind_bridge.cpp); cv::Mat ↔ numpy zero-copy via py::array_t; ScanEngine class exposing C++ to Python for real-time OCT processing
Infrastructure: Offline Logging	SHARPOfflineResilienceManager (src_python/sharp_resilience.py); SQLite offline telemetry buffer for patient scan logging in rural/disconnected clinics; Background heartbeat monitor; Auto-sync on reconnection

4. Export & Hardware Handoff
⚬ Export models to standard continuous ONNX format with dynamic batch sizes using torch.onnx.export.
⚬ Maintain verified numerical equivalence (tolerance ≤ 10⁻⁴) between PyTorch and ONNX Runtime execution.
⚬ Cross-compile C++ AnomalyCalculus and zero-copy bridges for embedded ARM processors (toolchain provided; not yet validated on hardware).
⚬ Provide calibration scripts for on-device clinical confidence readout and NMS clustering.
