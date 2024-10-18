# adapted from https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/cifar10-baseline.html # noqa: E501
import logging
from collections import OrderedDict

import lightning as L
import torch
import torch.nn.functional as F
import torchvision
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from torch import nn
from torchmetrics.functional import accuracy

logger = logging.Logger(__name__)


def create_model(
    num_classes: int,
) -> torchvision.models.VisionTransformer:

    model = torchvision.models.vit_b_16(
        weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
    )

    # modify head, swapping out linear layer
    model.heads = nn.Sequential(
        OrderedDict(
            [("head", nn.Linear(model.heads.head.in_features, num_classes))]
        )
    )

    return model


class ViTB16(L.LightningModule):
    def __init__(
        self,
        num_classes: int,
        freeze_embedding: bool = False,
        optimizer: OptimizerCallable = torch.optim.Adam,
        scheduler: LRSchedulerCallable = torch.optim.lr_scheduler.ReduceLROnPlateau,  # noqa: E501
        scheduler_config: dict | None = None,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model: torchvision.models.VisionTransformer = create_model(
            num_classes=num_classes,
        )
        self.num_classes = num_classes

        if freeze_embedding:
            # freeze embedding layer as per https://arxiv.org/abs/2211.09359
            self.model.conv_proj.requires_grad_(False)

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_config = scheduler_config or {}

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        model_output = self(x)
        loss = F.nll_loss(model_output, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        model_output = self(x)
        loss = F.nll_loss(model_output, y)
        preds = torch.argmax(model_output, dim=1)
        acc = accuracy(
            preds, y, task="multiclass", num_classes=self.num_classes
        )

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        scheduler = self.scheduler(optimizer)
        return (
            [optimizer],
            [{"scheduler": scheduler, **self.scheduler_config}],
        )
