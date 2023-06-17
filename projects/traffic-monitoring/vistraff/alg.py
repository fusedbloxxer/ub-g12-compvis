import cv2 as cv
import numpy as np
import typing as t
import pathlib as pb


class VehicleOccupancy(object):
    def __init__(self, masks_path: pb.Path, *args, **kwargs) -> None:
        super().__init__()
        self.lane_masks, self.lanes_mask = VehicleOccupancy.read_masks(masks_path)

    @staticmethod
    def read_masks(dir: pb.Path) -> t.Tuple[np.ndarray, np.ndarray]:
        masks: t.List[np.ndarray] = []

        for index in range(1, 10):
            mask: np.ndarray = cv.imread(str(dir / f'lane-{index}.png'), cv.IMREAD_GRAYSCALE)
            mask = (255 * mask.astype(bool)).astype(np.uint8)
            masks.append(mask)

        lane_masks: np.ndarray = np.stack(masks, axis=0)
        lanes_mask: np.ndarray = (255 * np.sum(lane_masks, axis=0).astype(bool)).astype(np.uint8)
        return lane_masks, lanes_mask
