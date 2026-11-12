# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightning integration for DeSTSeg."""

from pathlib import Path

import torch
from torch.nn import functional
from torchvision.transforms.v2 import functional as tvf

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.data.utils.generators.perlin import PerlinAnomalyGenerator
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

from .torch_model import DestsegModel


class Destseg(AnomalibModule):
    """Segmentation-guided denoising student-teacher anomaly detection.

    Args:
        anomaly_source_path: Optional directory of texture images for synthetic anomalies.
        student_steps: Number of student training steps before segmentation training.
        pretrained: Load ImageNet weights for the teacher.
        pre_processor: Pre-processor or default flag.
        post_processor: Post-processor or default flag.
        evaluator: Evaluator or default flag.
        visualizer: Visualizer or default flag.
    """

    def __init__(
        self,
        anomaly_source_path: Path | str | None = None,
        student_steps: int = 1000,
        pretrained: bool = True,
        pre_processor: PreProcessor | bool = True,
        post_processor: PostProcessor | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
    ) -> None:
        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )
        if student_steps < 0:
            msg = "student_steps must be non-negative"
            raise ValueError(msg)
        self.student_steps = student_steps
        self.model = DestsegModel(pretrained=pretrained)
        self.augmenter = PerlinAnomalyGenerator(anomaly_source_path=anomaly_source_path)

    def training_step(self, batch: Batch, *args, **kwargs) -> torch.Tensor:
        """Train the student first, then train the segmentation network."""
        del args, kwargs
        assert batch.image is not None
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        images = tvf.normalize(batch.image, [-m / s for m, s in zip(mean, std, strict=True)], [1 / s for s in std])
        augmented, mask = self.augmenter(images)
        augmented = tvf.normalize(augmented, mean, std)
        student_stage = self.global_step < self.student_steps
        self.model.student.train(student_stage)
        self.model.decoder.train(student_stage)
        if student_stage:
            _, student = self.model.features(augmented)
            self.model.teacher.eval()
            with torch.no_grad():
                teacher = [functional.normalize(feature, dim=1) for feature in self.model.teacher(batch.image)]
            loss = sum((1 - (t * s).sum(dim=1)).mean() for t, s in zip(teacher, student, strict=True))
        else:
            with torch.no_grad():
                teacher, student = self.model.features(augmented)
            prediction = self.model.segment(teacher, student)
            target = (
                functional.interpolate(mask.float(), size=prediction.shape[-2:], mode="bilinear", align_corners=False)
                >= 0.5
            ).float()
            probability = prediction * target + (1 - prediction) * (1 - target)
            loss = (
                functional.binary_cross_entropy(prediction, target, reduction="none") * (1 - probability).pow(4)
            ).mean()
            loss = loss + functional.l1_loss(prediction, target)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch.image.shape[0])
        return loss

    def validation_step(self, batch: Batch, *args, **kwargs) -> Batch:
        """Add image scores and anomaly maps to the batch."""
        del args, kwargs
        return batch.update(**self.model(batch.image)._asdict())

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure learning rates for student and segmentation components."""
        return torch.optim.SGD(
            [
                {"params": list(self.model.student.parameters()) + list(self.model.decoder.parameters()), "lr": 0.4},
                {"params": self.model.segment_residual.parameters(), "lr": 0.1},
                {
                    "params": list(self.model.segment_branches.parameters())
                    + list(self.model.segment_pool.parameters())
                    + list(self.model.segment_head.parameters()),
                    "lr": 0.01,
                },
            ],
            momentum=0.9,
            weight_decay=1e-4,
        )

    @property
    def trainer_arguments(self) -> dict:
        """Default training duration and validation settings."""
        return {"max_steps": 5000, "num_sanity_val_steps": 0}

    @property
    def learning_type(self) -> LearningType:
        """DeSTSeg learns from normal images."""
        return LearningType.ONE_CLASS
