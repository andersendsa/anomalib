# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for DeSTSeg training and inference."""

from types import SimpleNamespace

import pytest
import torch
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader

from anomalib import LearningType
from anomalib.data import ImageBatch
from anomalib.models import Destseg, get_model


@pytest.mark.parametrize("name", ["Destseg", "destseg"])
def test_registration(name: str) -> None:
    """The model can be constructed by its public name."""
    model = get_model(name, pretrained=False)
    assert isinstance(model, Destseg)
    assert model.learning_type == LearningType.ONE_CLASS
    assert isinstance(model.trainer_arguments, dict)


@pytest.mark.parametrize("student_steps", [0, 1])
def test_training_and_inference(student_steps: int) -> None:
    """Both training phases backpropagate and inference returns image and pixel scores."""
    model = Destseg(pretrained=False, student_steps=student_steps, pre_processor=False)
    images = torch.rand(2, 3, 64, 64)
    model.train()
    loss = model.training_step(SimpleNamespace(image=images))
    assert torch.isfinite(loss)
    loss.backward()
    assert all(parameter.grad is None for parameter in model.model.teacher.parameters())
    trained = model.model.student if student_steps else model.model.segment_head
    assert any(parameter.grad is not None for parameter in trained.parameters())
    model.eval()
    with torch.no_grad():
        prediction = model.model(images)
    assert prediction.pred_score.shape == (2,)
    assert prediction.anomaly_map.shape == (2, 1, 64, 64)


def test_trainer_switches_training_phase() -> None:
    """Lightning advances from student to segmentation training."""
    model = Destseg(
        pretrained=False,
        student_steps=1,
        pre_processor=False,
        post_processor=False,
        evaluator=False,
        visualizer=False,
    )
    batch = ImageBatch(image=torch.rand(2, 3, 64, 64))
    trainer = Trainer(max_steps=2, logger=False, enable_checkpointing=False, enable_progress_bar=False)
    trainer.fit(model, train_dataloaders=DataLoader([batch, batch], batch_size=None))
    assert trainer.global_step == 2
    assert any(parameter.grad is not None for parameter in model.model.segment_head.parameters())
