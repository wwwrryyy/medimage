"""
mobilenet_v3.py
===============
Stage 3 Pathology Classifier: MobileNetV3-Small on fixed 64x64 grayscale ROI crops.

- Depthwise-separable convolutions (torchvision MobileNetV3-Small backbone, trained from scratch).
- Single input channel (grayscale OCT/ultrasound) and 4 output classes (CNV, DME, DRUSEN, NORMAL).
- Temperature scaling (Guo et al. 2017): a single scalar T fitted on held-out validation
  logits so that softmax(logits / T) gives calibrated clinical confidence.
"""

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small

NUM_CLASSES = 4


class MobileNetV3Classifier(nn.Module):
    """MobileNetV3-Small producing raw class logits for [B, 1, 64, 64] inputs."""

    def __init__(self, num_classes: int = NUM_CLASSES, in_channels: int = 1):
        super().__init__()
        backbone = mobilenet_v3_small(weights=None, num_classes=num_classes)

        # Adapt the stem to grayscale input.
        stem = backbone.features[0][0]
        backbone.features[0][0] = nn.Conv2d(
            in_channels,
            stem.out_channels,
            kernel_size=stem.kernel_size,
            stride=stem.stride,
            padding=stem.padding,
            bias=False,
        )
        self.backbone = backbone
        # Temperature is a buffer so it is saved/loaded with the state dict.
        self.register_buffer("temperature", torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Calibrated class probabilities: softmax(logits / temperature)."""
        return F.softmax(self.forward(x) / self.temperature, dim=1)


class CalibratedClassifier(nn.Module):
    """Export wrapper: [B, 1, 64, 64] -> temperature-scaled class probabilities [B, C]."""

    def __init__(self, model: MobileNetV3Classifier):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.model(x) / self.model.temperature, dim=1)


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = 100,
) -> float:
    """
    Fits the scalar temperature T > 0 minimising negative log-likelihood of
    softmax(logits / T) on held-out data (LBFGS over log T keeps T positive).
    """
    if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.shape[0]:
        raise ValueError("logits must be [N, C] and labels [N] with matching N")

    logits = logits.detach().float()
    labels = labels.detach().long()
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().item())


def get_classifier(num_classes: int = NUM_CLASSES, in_channels: int = 1, **kwargs) -> MobileNetV3Classifier:
    """Factory returning the Stage 3 MobileNetV3 pathology classifier."""
    return MobileNetV3Classifier(num_classes=num_classes, in_channels=in_channels)
