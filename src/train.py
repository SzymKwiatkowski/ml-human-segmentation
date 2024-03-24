from pathlib import Path
import argparse
import yaml
import os

import lightning.pytorch as pl
from lightning.pytorch import loggers

from datamodules.segmentation import SegmentationDataModule
from models.segmentation_model import SegmentationModel


def load_config(path: Path) -> dict:
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def train(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config_file = args.config
    max_epochs = args.epochs
    use_neptune = args.use_neptune
    data_root_dir = args.data_root
    train_size = args.train_size
    batch_size = args.batch_size
    num_workers = args.workers
    patience = args.patience

    logger = None
    if use_neptune:
        config = load_config(config_file)
        token = config['config']['NEPTUNE_API_TOKEN']
        project = config['config']['PROJECT_NAME']
        logger = loggers.NeptuneLogger(
            project=project,
            api_token=token)
    else:
        logger = loggers.TensorBoardLogger(
            save_dir=Path('logs')
        )

    pl.seed_everything(42, workers=True)
    precision = 32

    datamodule = SegmentationDataModule(
        data_path=Path(data_root_dir),
        images_path=Path('Training_Images'),
        masks_path=Path('Ground_Truth'),
        batch_size=batch_size,
        num_workers=num_workers,
        train_size=train_size
    )

    model = SegmentationModel(
        encoder_name='resnet18',
        encoder_weights='imagenet',
        in_channels=3,
        out_channels=1,
        min_lr=1e-6,
        factor=0.8,
        patience=5,
        lr=5e-4
    )

    model.hparams.update(datamodule.hparams)

    model_summary_callback = pl.callbacks.ModelSummary(max_depth=-1)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(filename='{epoch}-{val_loss:.5f}', mode='min',
                                                       monitor='val_loss', verbose=True, save_last=True)
    early_stop_callback = pl.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=patience)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        logger=logger,
        devices=1,
        callbacks=[model_summary_callback, checkpoint_callback, early_stop_callback, lr_monitor],
        accelerator='cuda',
        strategy='ddp',
        precision=precision,
        max_epochs=max_epochs,
        log_every_n_steps=2
    )

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)

    print(f"Best model saved in: {checkpoint_callback.best_model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help')
    parser.add_argument('-c', '--config', action='store', default='config.yaml')
    parser.add_argument('-e', '--epochs', action='store', default=50,
                        type=int, help='Specified number of maximum epochs')
    parser.add_argument('-n', '--use-neptune', action='store', default=False,
                        type=bool, help='Use neptune as logger')
    parser.add_argument('-d', '--data-root', action='store', default='data')
    parser.add_argument('-b', '--batch-size', action='store', type=int, default=16)
    parser.add_argument('-p', '--patience', action='store', type=int, default=20)
    parser.add_argument('-w', '--workers', action='store', type=int, default=4)
    parser.add_argument('-t', '--train-size', action='store', type=float, default=0.6)

    args_parsed = parser.parse_args()
    train(args_parsed)
