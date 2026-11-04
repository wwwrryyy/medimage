"""
models/classifier/scan_classifier.py
====================================
Downstream Family (b): Unified MobileNetV3 scan classifier with Class Activation
Mapping (CAM) for simultaneous scan-level pathology diagnosis and spatial lesion localization.

Pre-registered in EXPERIMENT_PLAN.md:
- Full scan input: [B, 1, H, W] (e.g. 256x256 grayscale retinal OCT scan).
- Pathology classes: CNV (0), DME (1), DRUSEN (2), NORMAL (3).
- Primary localization map: CAM upsampled to full input resolution [B, H, W] normalized to [0, 1].
- Temperature scaling: scalar buffer T fitted on held-out validation logits.
"""

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small

from models.classifier.mobilenet_v3 import NUM_CLASSES, fit_temperature


class UnifiedScanClassifier(nn.Module):
    """
    Downstream Family (b) Unified MobileNetV3-Small scan classifier.
    Produces raw logits / calibrated probabilities and extracts Class Activation Maps (CAM)
    via vectorized Grad-CAM across the final convolutional layer.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, in_channels: int = 1):
        super().__init__()
        backbone = mobilenet_v3_small(weights=None, num_classes=num_classes)

        # Adapt stem conv for single-channel grayscale OCT
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
        self.num_classes = num_classes
        self.register_buffer("temperature", torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes raw logits [B, num_classes] for scan tensor [B, 1, H, W]."""
        return self.backbone(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Calibrated class probabilities: softmax(logits / temperature)."""
        return F.softmax(self.forward(x) / self.temperature, dim=1)

    def compute_cam(
        self,
        x: torch.Tensor,
        target_class: Optional[Union[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Computes Class Activation Maps (CAM) via vectorized Grad-CAM over the final conv layer.
        Upsamples the activation map to the input spatial dimensions [B, H, W] and normalizes to [0, 1].

        Args:
            x: Input scan tensor [B, 1, H, W] in [0.0, 1.0].
            target_class: Target class index (int) or per-batch tensor [B]. If None, uses top predicted class.

        Returns:
            torch.Tensor: Normalized CAM localization map [B, H, W] in [0.0, 1.0].
        """
        was_training = self.training
        self.eval()

        b_size, _, height, width = x.shape
        activations = []

        def forward_hook(module, inp, out):
            activations.append(out)

        hook_handle = self.backbone.features[-1].register_forward_hook(forward_hook)

        try:
            with torch.enable_grad():
                x_var = x.clone().detach().requires_grad_(True)
                features = self.backbone.features(x_var)
                pooled = self.backbone.avgpool(features)
                flattened = torch.flatten(pooled, 1)
                logits = self.backbone.classifier(flattened)

                if target_class is None:
                    targets = logits.argmax(dim=1)
                elif isinstance(target_class, int):
                    targets = torch.full((b_size,), target_class, dtype=torch.long, device=x.device)
                else:
                    targets = target_class.to(device=x.device, dtype=torch.long)

                feature_maps = activations[0]
                batch_indices = torch.arange(b_size, device=x.device)
                total_target_score = logits[batch_indices, targets].sum()

                grads = torch.autograd.grad(total_target_score, feature_maps)[0]
                weights = grads.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]

                cam = F.relu((weights * feature_maps).sum(dim=1, keepdim=True))  # [B, 1, H_f, W_f]
                cam_upsampled = F.interpolate(
                    cam, size=(height, width), mode="bilinear", align_corners=False
                ).squeeze(1)  # [B, H, W]

                cam_min = cam_upsampled.amin(dim=(1, 2), keepdim=True)
                cam_max = cam_upsampled.amax(dim=(1, 2), keepdim=True)
                range_diff = cam_max - cam_min

                normalized_cam = torch.where(
                    range_diff > 1e-8,
                    (cam_upsampled - cam_min) / (range_diff + 1e-8),
                    torch.zeros_like(cam_upsampled),
                )
        finally:
            hook_handle.remove()
            if was_training:
                self.train()

        return normalized_cam.detach()


class CalibratedScanClassifier(nn.Module):
    """Export wrapper for ONNX: [B, 1, H, W] -> temperature-scaled probabilities [B, C]."""

    def __init__(self, model: UnifiedScanClassifier):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.model(x) / self.model.temperature, dim=1)
