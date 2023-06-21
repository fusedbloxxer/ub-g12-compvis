import pathlib as pb
import typing as t
import cv2 as cv
import numpy as np


class Assets(object):
    def __init__(self, assets_path: pb.Path) -> None:
        super().__init__()

        # Premade assets
        self.lane_masks, self.lanes_mask = Assets.read_masks(assets_path / 'masks')
        self.bg_static: np.ndarray = Assets.read_static_bgs(assets_path / 'backgrounds')
        self.weights_path: pb.Path = assets_path / 'weights'
        self.trackers_path: pb.Path = assets_path / 'trackers'
        self.trackers: t.Dict[str, pb.Path] = {
            'bytetrack': self.trackers_path / 'bytetrack.yaml',
        }

        # Dataset Info
        self.image_width = 1920
        self.image_height = 880

        # Precompute the positions of the lane_masks, lanes_mask (N, 2)
        Y_coords: np.ndarray = np.arange(self.image_height)
        X_coords: np.ndarray = np.arange(self.image_width)
        grids: t.List[np.ndarray] = np.meshgrid(X_coords, Y_coords)
        X_matrix: np.ndarray = grids[0]
        Y_matrix: np.ndarray = grids[1]
        self.image_pos_grid: np.ndarray = np.stack((Y_matrix, X_matrix), axis=-1)

        # Define the lanes
        self.zone_all: np.ndarray = self.image_pos_grid[self.lanes_mask == 255]
        self.zone_a: np.ndarray = self.image_pos_grid[(self.lane_masks[:3] == 255).sum(axis=0).astype(bool)]
        self.zone_b: np.ndarray = self.image_pos_grid[(self.lane_masks[3:6] == 255).sum(axis=0).astype(bool)]
        self.zone_c: np.ndarray = self.image_pos_grid[(self.lane_masks[6:] == 255).sum(axis=0).astype(bool)]
        self.zones: t.List[np.ndarray] = [self.image_pos_grid[lane_mask == 255] for lane_mask in self.lane_masks]

    @staticmethod
    def read_static_bgs(directory: pb.Path) -> np.ndarray:
        bgs: t.List[np.ndarray] = []

        for index in range(1, 16):
            bg: np.ndarray = cv.imread(str(directory / f'context_{index}.png'))
            bg = cv.cvtColor(bg, cv.COLOR_BGR2RGB)
            bgs.append(bg)

        return np.stack(bgs, axis=0)

    @staticmethod
    def read_masks(directory: pb.Path) -> t.Tuple[np.ndarray, np.ndarray]:
        masks: t.List[np.ndarray] = []

        for index in range(1, 10):
            mask: np.ndarray = cv.imread(str(directory / f'lane-{index}.png'), cv.IMREAD_GRAYSCALE)
            mask = (255 * mask.astype(bool)).astype(np.uint8)
            masks.append(mask)

        lane_masks: np.ndarray = np.stack(masks, axis=0)
        lanes_mask: np.ndarray = (255 * np.sum(lane_masks, axis=0).astype(bool)).astype(np.uint8)
        return lane_masks, lanes_mask
