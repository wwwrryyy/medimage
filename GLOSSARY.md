# Point-of-Care Medical Imaging Diagnostics

Real-time optical coherence tomography (OCT) speckle suppression, unsupervised tissue baseline reconstruction, and spatial anomaly detection for portable edge devices.

## Language

**Scan Session**:
A continuous diagnostic acquisition period from a video or probe stream during which incoming frames are sequentially enhanced and analyzed for tissue anomalies.
_Avoid_: Stream loop, battlefield engine, pipeline run

**Frame**:
A single raw grayscale or color 2D slice captured from an OCT imaging probe prior to enhancement.
_Avoid_: Image, snapshot, tensor

**Despeckled Frame**:
A frame whose multiplicative speckle noise has been suppressed to reveal underlying anatomical structure.
_Avoid_: Clean image, filtered tensor, denoised frame

**Reconstruction**:
The healthy tissue baseline estimate produced by passing a despeckled frame through an unsupervised normal-tissue model.
_Avoid_: Autoencoder output, baseline frame, predicted image

**Anomaly Heatmap**:
A spatial map of pixel-wise reconstruction error between a despeckled frame and its healthy tissue reconstruction, highlighting potential pathology.
_Avoid_: Error matrix, diff map, threat matrix

**Finding**:
A contiguous anomalous tissue region isolated from an anomaly heatmap, characterized by its bounding box, reconstruction error score, size category, and diagnostic confidence.
_Avoid_: Threat, tactical alert, detection, candidate box
