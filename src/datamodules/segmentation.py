from pathlib import Path
from typing import Optional

from lightning import pytorch as pl
from torch.utils.data import DataLoader

from datasets.segmentation import SegmentationDataset
from datamodules.dataset_split import DatasetSplits
from datasets.dataset_transformations import DatasetTransformations


class SegmentationDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_path: Path,
            images_path: Path,
            masks_path: Path,
            shape: tuple = (320, 320),
            train_size: float = 0.6,
            batch_size: int = 32,
            num_workers: int = 4):
        super().__init__()

        self._data_path = data_path
        self._images_path = images_path
        self._masks_path = masks_path
        self._batch_size = batch_size
        self._train_size = train_size
        self._num_workers = num_workers
        self._shape = shape

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.save_hyperparameters(ignore=['data_path', 'number_of_workers', 'images_path', 'masks_path'])

    def setup(self, stage: Optional[str] = None):
        if self.train_dataset is not None:
            return

        images_path = self._data_path / self._images_path
        masks_path = self._data_path / self._masks_path
        images = list(images_path.glob('*.jpg'))

        train_images, val_images, test_images = DatasetSplits.basic_split(images, train_size=self._train_size)

        images_num = len(images)
        masks_num = len(list(masks_path.glob('*.png')))

        print('-' * 20)
        print(f'Images: {images_num}')
        print(f'Masks: {masks_num}')
        print('-' * 20)

        print(f'Train set: {len(train_images)}')
        print(f'Val set: {len(val_images)}')
        print(f'Test set: {len(test_images)}')
        print('-' * 25)

        self.train_dataset = SegmentationDataset(
            train_images,
            masks_path,
            self._shape,
            DatasetTransformations.augmentations()
        )

        self.val_dataset = SegmentationDataset(
            val_images,
            masks_path,
            self._shape,
            DatasetTransformations.transformations()
        )

        self.test_dataset = SegmentationDataset(
            test_images,
            masks_path,
            self._shape,
            DatasetTransformations.transformations()
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self._batch_size, num_workers=self._num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self._batch_size, num_workers=self._num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self._batch_size, num_workers=self._num_workers,
        )
