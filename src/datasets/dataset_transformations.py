import albumentations as A
from albumentations.pytorch import ToTensorV2


class DatasetTransformations:
    @staticmethod
    def augmentations():
        return A.Compose(
            [
              A.HorizontalFlip(),
              A.OneOf([A.GaussNoise(), A.GaussianBlur()]),
              A.Affine(rotate=5, shear=5),
              A.ElasticTransform(),
              A.RandomGridShuffle(grid=(8, 8), p=0.25),
              ToTensorV2(),
            ])

    @staticmethod
    def transformations():
        return A.Compose([
            ToTensorV2(),
        ])
