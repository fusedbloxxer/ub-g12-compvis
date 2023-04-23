import pathlib as pb
import typing as t
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


def same_cell_dim(grid_cells: t.List[t.List[np.ndarray]], dsize: t.Tuple[int, int]=(45, 45)) -> np.ndarray:
    cells: t.List[np.ndarray] = []
    for i in range(len(grid_cells)):
        for j in range(len(grid_cells[0])):
            grid_cell: np.ndarray = grid_cells[i][j]
            grid_cell = cv.resize(grid_cell, dsize=dsize, interpolation=cv.INTER_LINEAR)
            cells.append(grid_cell)
    return np.stack(cells, axis=0)

