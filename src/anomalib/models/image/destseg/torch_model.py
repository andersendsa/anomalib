# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DeSTSeg teacher, denoising student and segmentation network."""

import timm
import torch
from torch import nn
from torch.nn import functional
from torchvision.models.resnet import BasicBlock

from anomalib.data import InferenceBatch


def residual_layer(in_channels: int, out_channels: int) -> nn.Sequential:
    """Build two residual blocks with an optional projection."""
    projection = None
    if in_channels != out_channels:
        projection = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels))
    return nn.Sequential(
        BasicBlock(in_channels, out_channels, downsample=projection),
        BasicBlock(out_channels, out_channels),
    )


class DestsegModel(nn.Module):
    """Denoising student-teacher model with a multiscale segmentation head.

    Args:
        pretrained: Load ImageNet weights for the frozen teacher.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.teacher = timm.create_model("resnet18", pretrained=pretrained, features_only=True, out_indices=(1, 2, 3))
        self.teacher.requires_grad_(requires_grad=False)
        self.student = timm.create_model("resnet18", pretrained=False, features_only=True, out_indices=(1, 2, 3, 4))
        self.decoder = nn.ModuleList([residual_layer(a, b) for a, b in [(512, 512), (512, 256), (256, 128), (128, 64)]])
        self.segment_residual = residual_layer(448, 256)
        self.segment_branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(256, 256, 3, padding=d, dilation=d, bias=False), nn.BatchNorm2d(256), nn.ReLU())
            for d in (6, 12, 18)
        ])
        self.segment_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(256, 256, 1), nn.ReLU())
        self.segment_head = nn.Sequential(
            nn.Conv2d(1024, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 1, 1),
        )

    def features(self, images: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Extract normalized teacher and decoded student feature pyramids."""
        self.teacher.eval()
        with torch.no_grad():
            teacher = [functional.normalize(feature, dim=1) for feature in self.teacher(images)]
        encoded = self.student(images)
        decoded = self.decoder[0](encoded[3])
        student = []
        for layer, target in zip(self.decoder[1:], reversed(encoded[:3]), strict=True):
            decoded = layer(
                functional.interpolate(decoded, size=target.shape[-2:], mode="bilinear", align_corners=False),
            )
            student.append(functional.normalize(decoded, dim=1))
        return teacher, list(reversed(student))

    def segment(self, teacher: list[torch.Tensor], student: list[torch.Tensor]) -> torch.Tensor:
        """Predict anomaly probabilities from feature discrepancies."""
        size = teacher[0].shape[-2:]
        discrepancies = [
            functional.interpolate(-t * s, size=size, mode="bilinear", align_corners=False)
            for t, s in zip(teacher, student, strict=True)
        ]
        features = self.segment_residual(torch.cat(discrepancies, dim=1))
        pooled = functional.interpolate(self.segment_pool(features), size=size, mode="bilinear", align_corners=False)
        return self.segment_head(
            torch.cat([pooled, *(branch(features) for branch in self.segment_branches)], dim=1),
        ).sigmoid()

    def forward(self, images: torch.Tensor) -> InferenceBatch | tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Return feature pyramids for training, or anomaly predictions for evaluation."""
        teacher, student = self.features(images)
        if self.training:
            return teacher, student
        anomaly_map = functional.interpolate(
            self.segment(teacher, student),
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        flattened = anomaly_map.flatten(1)
        pred_score = flattened.topk(min(100, flattened.shape[1]), dim=1).values.mean(dim=1)
        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)
