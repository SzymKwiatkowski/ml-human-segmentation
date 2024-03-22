import numpy as np
from pathlib import Path
from PIL import Image


def load_image(file_path: Path, shape: tuple[int]) -> np.ndarray:
    return np.asarray(Image.open(file_path).resize(shape))
