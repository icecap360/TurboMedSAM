from typing import Union

import hiera
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import Accuracy
from torchvision.models import ResNet50_Weights, resnet50


class LitHiera(pl.LightningModule):
    def __init__(
        self,
        learning_rate=1e-3,
        model: Union[str, nn.Module] = "base",
        checkpoint="mae_in1k_ft_in1k",
        num_classes=1000,
    ):
        super().__init__()

        self.num_classes = num_classes
        if model == "base":
            self.hiera = hiera.hiera_base_224(pretrained=True, checkpoint=checkpoint)
        elif model == "tiny":
            self.hiera = hiera.hiera_tiny_224(pretrained=True, checkpoint=checkpoint)
        elif isinstance(model, nn.Module):
            self.hiera = model
        else:
            raise ValueError(
                f"Model {model} not recognized. Must be 'base', 'tiny', or nn.Module."
            )

        self.learning_rate = learning_rate

        self.loss_module = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)

        # Save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters(ignore="model")

    def forward(self, x):
        return self.hiera(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def _get_outputs_loss_accuracy(self, batch):
        inputs, targets = batch["pixel_values"], batch["labels"]
        outputs = self(inputs)
        loss = self.loss_module(outputs, targets)
        acc = self.accuracy(outputs, targets)

        return outputs, loss, acc

    def _log_step(self, stage, acc, loss):
        self.log(
            f"{stage}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{stage}_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

    def training_step(self, batch, batch_idx):
        _, loss, acc = self._get_outputs_loss_accuracy(batch)

        self._log_step("train", acc, loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs, loss, acc = self._get_outputs_loss_accuracy(batch)

        self._log_step("val", acc, loss)

        return outputs

    def test_step(self, batch, batch_idx):
        outputs, loss, acc = self._get_outputs_loss_accuracy(batch)

        self._log_step("test", acc, loss)

        # Return outputs for use in custom callback.
        return outputs

    @property
    def blocks_repr(self):
        stages_channels = []
        stages_channels.append(self.hiera.blocks[0].dim_out)
        stages_depths = [1]

        for i, blk in enumerate(self.hiera.blocks[1:]):
            if blk.dim_out != stages_channels[-1]:
                stages_channels.append(blk.dim_out)
                stages_depths.append(1)
            else:
                stages_depths[-1] += 1

        return [(c, d) for c, d in zip(stages_channels, stages_depths)]
