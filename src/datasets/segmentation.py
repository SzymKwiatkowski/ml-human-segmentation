import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SegmentationDataset(Dataset):
    def __init__(self, images, masks, shape, transform):
        super(SegmentationDataset, self).__init__()

        self._masks_path = masks
        self._images = images
        self._shape = shape

        self._transformations = transform

    def __getitem__(self, index):
        img_path = self._images[index]
        mask_path = self._masks_path / img_path.name.replace('jpg', 'png')

        image = np.asarray(Image.open(img_path).convert('RGB').resize(self._shape)) / 255.
        mask = np.asarray(Image.open(mask_path).convert('L').resize(self._shape)) // 255

        transformed = self._transformations(image=image, mask=mask)

        return transformed['image'].type(torch.float32), transformed['mask'][None, ...].type(torch.int)

    def __len__(self):
        return len(self._images)
