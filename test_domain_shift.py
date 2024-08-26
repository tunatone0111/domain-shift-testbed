from dataclasses import dataclass

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics.classification import MulticlassAccuracy

from datamodule import DomainShiftDataModule, DomainShiftDMConfig


@dataclass
class DomainShiftConfig(DomainShiftDMConfig):
    model: nn.Module
    log_dir: str
    num_classes: int


class DomainShiftTest(L.LightningModule):
    def __init__(self, cfg: DomainShiftConfig):
        super().__init__()
        self.model = cfg.model
        self.accuracy = MulticlassAccuracy(num_classes=cfg.num_classes, average="macro")
        self.loss_list = []

    def forward(self, x):
        return self.model(x)

    def test_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]

        logits = self.forward(x)

        loss = F.cross_entropy(logits, y, reduction="none")

        self.loss_list.append(loss)

        self.accuracy(logits, y)

        return loss.mean()

    def on_test_epoch_end(self):
        tensorboard = self.logger.experiment
        tensorboard.add_histogram("loss", torch.concat(self.loss_list), 0)

        self.log("accuracy", self.accuracy.compute())


def test_domain_shift(cfg: DomainShiftConfig):
    dm = DomainShiftDataModule(cfg=cfg)
    module = DomainShiftTest(cfg=cfg)

    trainer = L.Trainer(
        default_root_dir=cfg.log_dir,
        devices=[1],
    )
    trainer.test(model=module, datamodule=dm)
