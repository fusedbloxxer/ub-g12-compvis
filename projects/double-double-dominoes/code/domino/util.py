import pathlib as pb
import numpy as np
import cv2 as cv


def folder_exists(path: pb.Path) -> bool:
    if not path.is_dir():
        raise NotADirectoryError(path.absolute())
    if not path.exists():
        raise FileNotFoundError(path.absolute())
    return True


def keep_aspect_ratio(image: np.ndarray, size: np.ndarray) -> np.ndarray:
    if np.array_equal(image.shape, size.shape):
        return image
    if np.array_equal(np.flip(image.shape), size):
        image_transposed: np.ndarray = cv.transpose(image)
        image_flipped: np.ndarray = cv.flip(image_transposed, 1)
        return image_flipped.reshape(size)
    return image
