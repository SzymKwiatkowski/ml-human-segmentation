import lightning.pytorch as pl
import segmentation_models_pytorch as smp
import torch
from monai.losses import DiceLoss
from torchmetrics import Metric, MetricCollection, F1Score, Precision, Recall, Dice


class SegmentationModel(pl.LightningModule):
    def __init__(self, encoder_name, encoder_weights, in_channels, out_channels, **kwargs):
        super().__init__()

        activation_function = 'sigmoid' if out_channels == 1 else 'softmax'
        self.activation = torch.nn.Sigmoid() if out_channels == 1 else torch.nn.Softmax(dim=1)

        self.network = smp.Unet(
          encoder_name=encoder_name,
          encoder_weights=encoder_weights,
          in_channels=in_channels,
          classes=out_channels,
          activation=activation_function,
        )

        self.loss = DiceLoss()

        task = 'binary'
        metrics = MetricCollection([
            MetricCollection(
                F1Score(task, threshold=0.5),
                Precision(task, threshold=0.5),
                Recall(task, threshold=0.5),
                Dice(threshold=0.5),
            ),
        ])

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
        self.log_dict(self.val_metrics, sync_dist=True)

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss(y_pred, y)

        self.log('test_loss', loss, on_epoch=True, sync_dist=True)
        self.test_metrics.update(y_pred, y)
        self.log_dict(self.test_metrics, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.trainer.model.parameters(), lr=3e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=True,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train_loss',
                'interval': 'step',
                'frequency': 1
            }
        }