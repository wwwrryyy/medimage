# Experiment Plan (Pre-registered)

**Status:** Pre-registered on 2026-10-30, before any experiment in this plan was run.
After the first M4 run, any change to the hypotheses, metrics or decision rules must be logged under *Deviations* with a reason.

## 1. Research question
Does a learned despeckling front-end improve downstream retinal OCT lesion detection and classification within an edge compute budget?

## 2. Hypotheses
| ID | Hypothesis | Supported if |
|---|---|---|
| **H1** | A learned despeckler improves downstream performance over raw input, at lower latency than BM3D. | For **both** downstream families, the best learned front-end beats Raw on macro AUROC (Kermany re-split test) **or** AUPRO (lesion set), with a paired patient-bootstrap 95% CI that excludes 0. Its single-thread latency must also be below BM3D's. |
| **H2** | Despeckling reduces the performance drop across scanner vendors. | Cross-vendor drop = (Spectralis score − non-Spectralis score) on the lesion set. The drop with the best learned front-end is smaller than with Raw, and the paired bootstrap CI of the difference excludes 0. |

Secondary (exploratory, not confirmatory): does PSNR/SSIM on Duke 2012 rank front-ends in the same order as downstream utility? Measured with Spearman ρ.

## 3. Data
| Set | Use | Notes |
|---|---|---|
| Kermany OCT2017 (full, ~84k) | Train/val/test for classification | New patient-level re-split 80/10/10, stratified by class. Duplicate images found by perceptual hash are merged within one patient group before splitting. |
| Kermany official test (1,000) | Literature comparison only | Never used for model selection or headline claims |
| RETOUCH | Lesion-level eval (IRF/SRF/PED fluid) + N2N training volumes | Primary. Split by volume so N2N training volumes ≠ eval volumes. Results reported per vendor. |
| Duke DME 2015 | Lesion-level eval fallback (fluid) | Used until RETOUCH access arrives. Reported separately. |
| Duke SD-OCT 2012 | Restoration metrics (PSNR/SSIM) | Exploratory only |

Drusen are evaluated at scan level only (the lesion sets don't annotate drusen).
Normal-only subsets for the autoencoder come from the training split only.

## 4. Experimental arms
**Front-ends (5):** `raw`, `nlm`, `bm3d`, `unet_synth` (current ConvUNet), `unet_n2n` (Neighbor2Neighbor ConvUNet trained on adjacent B-scans from RETOUCH/Duke training volumes, same architecture as `unet_synth`).

**Downstream families (2):**
- **(a)** ConvAutoencoder → AnomalyCalculus → ROIDecisionLogic NMS → MobileNetV3 crop classifier. The classifier is retrained on AE-proposed crops from training scans, using MIL (each scan is a bag with the scan's label).
- **(b)** Unified MobileNetV3 scan classifier, with CAM as the localization map.

**Protocol:** The front-end is frozen. Downstream models are retrained from scratch for each front-end (5 × 2 × 3 seeds = 30 runs).
**Secondary arm:** `unet_n2n` fine-tuned end-to-end with family (b)'s loss (3 seeds). It is reported separately and never counted toward H1.

Hyperparameters are tuned once, on `raw` + validation set, then held fixed across all front-ends.

## 5. Metrics
| Kind | Metric | Set |
|---|---|---|
| Classification | Macro AUROC (primary), ECE (15 bins) | Kermany re-split test; official test for comparison |
| Localization | AUPRO up to FPR 0.3 (primary), Dice at the val-chosen threshold | RETOUCH / Duke DME fluid masks |
| Efficiency | MACs, params, single-thread ONNX Runtime latency (FP32 + INT8, Apple M5 Mac CPU, median of 200 runs after 20 warm-up) | Front-end + downstream, measured separately and end-to-end |
| Exploratory | PSNR, SSIM | Duke 2012 |

Family (a) localization map = anomaly heatmap. Family (b) localization map = CAM upsampled to input size.

## 6. Statistics
- 3 seeds per configuration; report mean ± std across seeds.
- 95% CIs by patient-level (Kermany) or volume-level (RETOUCH) bootstrap, 2,000 resamples.
- Front-end comparisons are paired (same patients/volumes in each resample). Holm correction across the 4 non-raw front-ends within each family.
- No early stopping on test sets; checkpoints are selected on validation macro AUROC only.

## 7. Runner & results
- One YAML config per run in `experiments/configs/`; `experiments/run.py <config>` trains and evaluates.
- Each run appends a row to `experiments/results.csv`: `run_id, git_sha, front_end, family, seed, split, metric, value`.
- Analysis (bootstrap, Holm, tables) lives in `experiments/analyze.py`, which reads only `results.csv`.

## 8. Out of scope (frozen)
The C++ engine, SHARP offline logging and HIPAA anonymizer are frozen: tests stay green, no new features until M5.

## 9. Stretch goals (only after M4)
- Conformal-prediction referral output (guaranteed-coverage "refer / no-refer").
- RETFound-teacher distillation into MobileNetV3 as an extra arm.

## 10. Milestones
M0 doc fixes → M1 data → M2 eval harness + baselines on `raw` → M3 despeckler arms incl. N2N → M4 full grid + stats → M5 write-up.

## Deviations
1. **BM3D Training Computation**: Classical BM3D filtering requires ~1,200 ms per frame on CPU (< 1 FPS). Running on-the-fly BM3D across 6,652 training scans for 15 epochs across 6 seeds would require ~199 CPU hours. Per `data_scripts/make_targets.py`, BM3D is therefore retained strictly as an inference-time/val-time restoration and benchmark comparison, while fast Non-Local Means (NLM) serves as the primary classical spatial filtering baseline for full training.
2. **Kermany Dataset Partition**: Training and evaluation were conducted on a verified patient-isolated 7,770-scan split (with DSU clustering and perceptual dHash deduplication) to eliminate patient leakage while remaining within edge compute constraints.

