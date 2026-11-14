# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""VerdictAD: few-shot anomaly detection via decision-guided refinement."""

from typing import NoReturn

import torch
from torchvision.transforms.v2 import InterpolationMode, Resize

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule, MemoryBankMixin
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

from .torch_model import VerdictADModel


class VerdictAD(MemoryBankMixin, AnomalibModule):
    """Refine AnomalyDINO predictions using global and local visual decisions.

    The first ``n_shots`` training images supply both the detector's memory bank
    and Clef-Flash's references. Use normal-only training data from one category.
    This implements AnomalyDINO + VerdictAD, not the paper's separate DINOv2-L baseline.

    Args:
        n_shots (int): Number of normal references, one to four. Defaults to four.
        encoder_name (str): AnomalyDINO backbone. Defaults to DINOv2-L.
        decision_model_path (str | None): Trusted local Clef-Flash release, required
            for refinement. Its Python code is executed on first inference.
        decision_batch_size (int): Number of decision records per forward pass.
        alpha (float): Global refinement strength; zero disables GVR.
        gamma (float): Local refinement strength; zero disables LVR.
        pre_processor (PreProcessor | bool): Resize-only preprocessor or enable flag.
        post_processor (PostProcessor | bool): Calibration/thresholding or enable flag.
        evaluator (Evaluator | bool): Evaluator or enable flag.
        visualizer (Visualizer | bool): Visualizer or enable flag.
    """

    def __init__(
        self,
        n_shots: int = 4,
        encoder_name: str = "vit_large_patch14_dinov2",
        decision_model_path: str | None = None,
        decision_batch_size: int = 4,
        alpha: float = 0.5,
        gamma: float = 0.25,
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
        self.model = VerdictADModel(n_shots, encoder_name, decision_model_path, decision_batch_size, alpha, gamma)

    def training_step(self, batch: Batch, *args, **kwargs) -> torch.Tensor:
        """Collect normal reference images and return a dummy Lightning loss.

        Args:
            batch (Batch): Normal training images.
            *args: Unused positional arguments.
            **kwargs: Unused keyword arguments.

        Returns:
            torch.Tensor: Dummy differentiable scalar loss.

        Raises:
            ValueError: If labeled anomalous training images are supplied.
        """
        del args, kwargs
        if batch.gt_label is not None and batch.gt_label.any():
            msg = "VerdictAD training references must all be normal."
            raise ValueError(msg)
        self.model(batch.image)
        return torch.tensor(0.0, requires_grad=True, device=self.device)

    def fit(self) -> None:
        """Fit the detector on the collected references."""
        self.model.fit()

    def validation_step(self, batch: Batch, *args, **kwargs) -> Batch:
        """Refine predictions before calibration and evaluation.

        Args:
            batch (Batch): Query images and optional ground truth.
            *args: Unused positional arguments.
            **kwargs: Unused keyword arguments.

        Returns:
            Batch: Batch containing refined scores and maps.
        """
        del args, kwargs
        predictions = self.model(batch.image)
        return batch.update(**predictions._asdict())

    @classmethod
    def configure_pre_processor(cls, image_size: tuple[int, int] | int | None = None) -> PreProcessor:
        """Preserve RGB values for the decision model; normalize inside the detector.

        Args:
            image_size (tuple[int, int] | int | None): Target size, default ``(896, 896)``.

        Returns:
            PreProcessor: Bilinear resize without ImageNet normalization.
        """
        return PreProcessor(Resize(image_size or (896, 896), interpolation=InterpolationMode.BILINEAR, antialias=True))

    @staticmethod
    def configure_optimizers() -> None:
        """Use no optimizer because both pretrained models are frozen."""
        return

    @property
    def trainer_arguments(self) -> dict[str, int]:
        """Return single-device, single-epoch trainer settings."""
        return {"max_epochs": 1, "num_sanity_val_steps": 0, "devices": 1}

    @property
    def learning_type(self) -> LearningType:
        """Return one-class learning with a bounded normal reference set."""
        return LearningType.ONE_CLASS

    def to_torch(self, *args, **kwargs) -> NoReturn:
        """Reject export; restore Lightning checkpoints with the external decision release instead."""
        del args, kwargs
        msg = f"{self.name} export is unsupported. Use a Lightning checkpoint and the local Clef-Flash release."
        raise NotImplementedError(msg)

    def to_onnx(self, *args, **kwargs) -> NoReturn:
        """Reject ONNX export of the PIL/custom-code decision pipeline."""
        self.to_torch(*args, **kwargs)

    def to_openvino(self, *args, **kwargs) -> NoReturn:
        """Reject OpenVINO export of the PIL/custom-code decision pipeline."""
        self.to_torch(*args, **kwargs)
