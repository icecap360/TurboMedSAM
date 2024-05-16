import pytorch_lightning as pl
import torch
import wandb
from datasets import load_dataset
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights, resnet50


# Define a custom collate function
def collate_fn(examples):
    images = []
    labels = []

    for example in examples:
        images.append((example["pixel_values"]))
        labels.append(example["label"])

    pixel_values = torch.stack(images)
    labels = torch.tensor(labels)

    return {"pixel_values": pixel_values, "labels": labels}


def input_transforms(examples, composed_transforms=None):
    examples["pixel_values"] = [
        composed_transforms(image.convert("RGB"))
        if composed_transforms
        else image.convert("RGB")
        for image in examples["image"]
    ]

    return examples


def resnet50_transforms(examples):
    composed_transforms = ResNet50_Weights.DEFAULT.transforms()
    return input_transforms(examples, composed_transforms)


class LogPredictionsCallback(Callback):
    def __init__(self, datamodule):
        super().__init__()
        self.dm = datamodule

    def log_images_table(self, trainer, outputs, wandb_logger, batch, n):
        x_key, y_key = batch
        x, y = batch[x_key][:n], batch[y_key][:n]
        outputs = outputs[:n].argmax(dim=-1)

        # Convert labels to strings
        label_str_fn = self.dm.dataset_val.features["label"].int2str
        y = label_str_fn(y)
        outputs = label_str_fn(outputs)

        # Log predictions as a Table
        columns = ["Image", "Ground Truth", "Prediction"]

        data = [
            [wandb.Image(x_i), y_i, y_pred]
            for x_i, y_i, y_pred in list(zip(x, y, outputs))
        ]
        wandb_logger.log_table(key="sample_table", columns=columns, data=data)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Called when the test batch ends."""
        wandb_logger = trainer.logger

        # Log 20 sample image predictions from first batch
        if batch_idx == 0:
            self.log_images_table(trainer, outputs, wandb_logger, batch, n=20)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Called when the validation batch ends."""

        wandb_logger = trainer.logger

        # Let's log 20 sample image predictions from first batch
        if batch_idx == 0:
            self.log_images_table(trainer, outputs, wandb_logger, batch, n=20)


class ImageNet1KDataModule(pl.LightningDataModule):
    def __init__(self, cache_dir, batch_size=128, num_workers=32):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_dir = cache_dir

    def prepare_data(self) -> None:
        # Download the dataset
        self.dataset_train = load_dataset(
            "imagenet-1k", split="train", cache_dir=self.cache_dir
        )

        self.dataset_val = load_dataset(
            "imagenet-1k", split="validation", cache_dir=self.cache_dir
        )

        # ImageNet-1K validation set is used as the test set since labels
        # are not publicly available for the test set.
        self.dataset_test = load_dataset(
            "imagenet-1k", split="validation", cache_dir=self.cache_dir
        )

    def setup(self, stage=None):
        # Assign train/val/test/predict datasets for use in dataloaders.
        # Apply augmentations to the train/val dataset.
        if stage == "fit":
            self.dataset_train = self.dataset_train.with_transform(resnet50_transforms)  # type: ignore
            self.dataset_val = self.dataset_val.with_transform(resnet50_transforms)  # type: ignore

        if stage == "test":
            # Apply the basic transform to RGB
            self.dataset_test = self.dataset_test.with_transform(resnet50_transforms)  # type: ignore

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=collate_fn,
        )
