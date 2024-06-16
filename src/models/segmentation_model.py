import lightning.pytorch as pl
import segmentation_models_pytorch as smp
import torch
from monai.losses import DiceLoss
from torchmetrics import MetricCollection, F1Score, Precision, Recall, Dice

import logging


class SegmentationModel(pl.LightningModule):
    def __init__(self,
                 factor,
                 lr,
                 min_lr,
                 patience,
                 encoder_name,
                 encoder_weights,
                 in_channels,
                 out_channels,
                 monitored_metric: str = 'train_loss',
                 lr_scheduler_interval: str = 'step',
                 lr_scheduler_freq: int = 1,
                 lr_scheduler_mode: str = 'min',
                 **kwargs):
        super().__init__()

        activation_function = 'sigmoid' if out_channels == 1 else 'softmax'
        self.activation = torch.nn.Sigmoid() if out_channels == 1 else torch.nn.Softmax(dim=1)
        self_logger = logging.getLogger(__name__)
        logging.basicConfig(filename='myapp.log', level=logging.INFO)
        if kwargs['model'] == "unet":
            self_logger.info("Selecting unet")
            self.network = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=out_channels,
                activation=activation_function,
                )
        elif kwargs['model'] == "unet++":
            self_logger.info("Selecting unet++")
            self.network = smp.UnetPlusPlus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=out_channels,
                activation=activation_function,
                )
        elif kwargs['model'] == "deeplabv3":
            self_logger.info("Selecting deeplab")
            self.network = smp.DeepLabV3(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=out_channels,
                activation=activation_function,
                )
        else:
            self_logger.info("Selecting unet")
            self.network = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=out_channels,
                activation=activation_function,
                )

        self._lr = lr
        self._factor = factor
        self._patience = patience
        self._min_lr = min_lr
        self._monitored_metric = monitored_metric
        self._lr_scheduler_interval = lr_scheduler_interval
        self._lr_scheduler_freq = lr_scheduler_freq
        self._lr_scheduler_mode = lr_scheduler_mode

        self.loss = DiceLoss()

        task = 'binary'
        metrics = MetricCollection(
            {
                "f1_score": F1Score(task, threshold=0.5),
                "precision": Precision(task, threshold=0.5),
                "recall": Recall(task, threshold=0.5),
                "dice_score": Dice(threshold=0.5),
            }
        )

        self.train_metrics = metrics.clone('train_')
        self.val_metrics = metrics.clone('val_')
        self.test_metrics = metrics.clone('test_')

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss(y_pred, y)

        self.log('train_loss', loss, sync_dist=True, prog_bar=True)
        self.train_metrics.update(y_pred, y)
        self.log_dict(self.train_metrics, sync_dist=True)

        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss(y_pred, y)

        self.log('val_loss', loss, on_epoch=True, sync_dist=True)
        self.val_metrics.update(y_pred, y)
        self.log_dict(self.val_metrics)

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss(y_pred, y)

        self.log('test_loss', loss, on_epoch=True, sync_dist=True)
        self.test_metrics.update(y_pred, y)
        self.log_dict(self.test_metrics)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.trainer.model.parameters(),
            lr=self._lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            amsgrad=True,
            weight_decay=1.25e-2)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=self._lr_scheduler_mode,
            factor=self._factor,
            patience=self._patience,
            min_lr=self._min_lr,
            verbose=True,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': self._monitored_metric,
                'interval': self._lr_scheduler_interval,
                'frequency': self._lr_scheduler_freq
            }
        }
