from typing import Any
import torch
import cv2 as cv
import numpy as np
import typing as t
import pathlib as pb

from .res import Assets
from .data import CarTrafficTaskOneDict, CarTrafficTaskTwoDict
from .model import FasterRCNNMOD, YOLOMOD


class VehicleOccupancy(object):
    def __init__(self, assets: Assets, device: torch.device, *args, **kwargs) -> None:
        super().__init__()
        self.device = device
        self.assets: Assets = assets

        # Initialize models
        self.model_frcnn = FasterRCNNMOD(box_score_thresh=0.25, box_nms_thresh=0.35)
        self.model_yolo_uadetrac = YOLOMOD(assets)
        self.model_yolo_pretraned = YOLOMOD()

        # Send models to the available device
        self.model_frcnn.model.to('cpu')
        self.model_yolo_uadetrac.model.to(device)
        self.model_yolo_pretraned.model.to(device)

        # Set models to evaluation mode
        self.model_frcnn.model.eval()
        self.model_yolo_uadetrac.model
        self.model_yolo_pretraned.model

        # Params
        self.min_bbox_area: float = 1000
        self.max_lane_distance: float = 12
        self.max_zone_distance: float = 50
        self.min_countour_area: float = 2_000.0
        self.max_contour_area: float = (1920. / 3) * (880. / 4)

    def bbox_mask(self, image: np.ndarray, bg: np.ndarray) -> t.Tuple[np.ndarray, np.ndarray]:
        # Reduce to grayscale
        bg_gray: np.ndarray = cv.cvtColor(bg.copy(), cv.COLOR_RGB2GRAY)
        image_gray: np.ndarray = cv.cvtColor(image.copy(), cv.COLOR_RGB2GRAY)

        # Compute foreground by using background subtraction
        fg_mask: np.ndarray = np.abs(image_gray.astype(np.int32) - bg_gray.astype(np.int32)).astype(np.uint8)        

        # Remove noise and apply Otsu's algorithm
        fg_mask = cv.GaussianBlur(fg_mask,(5,5),0)
        _, fg_mask = cv.threshold(fg_mask, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        # Apply erosion to filter out noisy points such as leaves
        kernel = cv.getStructuringElement(cv.MORPH_RECT, ksize=(3, 3))
        fg_mask = cv.erode(fg_mask, kernel, iterations=2)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, ksize=(5, 5))
        fg_mask = cv.erode(fg_mask, kernel, iterations=1)

        # Apply dilation to emphasize cars
        kernel = cv.getStructuringElement(cv.MORPH_CROSS, ksize=(3, 3))
        fg_mask = cv.dilate(fg_mask, kernel, iterations=3)
        kernel = cv.getStructuringElement(cv.MORPH_CROSS, ksize=(5, 5))
        fg_mask = cv.dilate(fg_mask, kernel, iterations=3)
        kernel = cv.getStructuringElement(cv.MORPH_CROSS, ksize=(7, 7))
        fg_mask = cv.dilate(fg_mask, kernel, iterations=2)
        kernel = cv.getStructuringElement(cv.MORPH_CROSS, ksize=(9, 9))
        fg_mask = cv.dilate(fg_mask, kernel, iterations=2)

        # Apply countours to find areas of interest
        contours, _ = cv.findContours(fg_mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        # Build a new mask to contain BBOXes of interest
        bbox_mask: np.ndarray = np.zeros_like(fg_mask, dtype=np.uint8)

        # Compute BBOXes from contours
        for cnt in contours:
            if self.min_countour_area < cv.contourArea(cnt) < self.max_contour_area:
                x, y, width, height = cv.boundingRect(cnt)
                bbox_mask[y: y + height, x: x + width] = 255

        # Compute masked image
        image_copy: np.ndarray = image.copy()
        image_masked = cv.bitwise_and(image_copy, image_copy, mask=bbox_mask)
        return bbox_mask, image_masked

    def distance(self, bboxes: np.ndarray, zone: np.ndarray, pts: str='center') -> np.ndarray:
        if pts == 'center':
            # Compute the distance from each BBOX to each zone coordinate
            bboxes_pts_xy: np.ndarray = np.column_stack(((bboxes[:, 0] + bboxes[:, 2]) / 2, (bboxes[:, 1] + bboxes[:, 3]) / 2))
        elif pts == 'a':
            # Compute the distance from each BBOX to each zone coordinate
            bboxes_pts_xy: np.ndarray = np.column_stack((0.25 * bboxes[:, 0] + 0.75 * bboxes[:, 2], bboxes[:, 3]))
        elif pts == 'b':
            # Compute the distance from each BBOX to each zone coordinate
            bboxes_pts_xy: np.ndarray = np.column_stack((bboxes[:, 0], bboxes[:, 3]))
        elif pts == 'c':
            # Compute the distance from each BBOX to each zone coordinate
            bboxes_pts_xy: np.ndarray = np.column_stack((0.25 * bboxes[:, 0] + 0.75 * bboxes[:, 2], bboxes[:, 3]))
        else:
            raise Exception('Invalid pts computation method.')

        # Compute distances
        bboxes_pts_yx: np.ndarray = np.flip(bboxes_pts_xy, axis=1)
        distances: np.ndarray = np.linalg.norm((bboxes_pts_yx[:, None, :] - zone), axis=2)

        # Compute the minimum distance
        min_zone_dist: np.ndarray = np.take_along_axis(distances, distances.argmin(axis=1)[..., None], axis=1)

        # Indicate BBOXes outside that zone
        return min_zone_dist

    def __call__(self, data: CarTrafficTaskOneDict) -> t.List[int]:
        image: np.ndarray = data['image'].copy()

        # Retrive GIMP made background image
        bg: np.ndarray = self.assets.bg_static[data['context'] - 1]

        # Compute mask containing BBOXes with possible changes in the image from precomputed background
        bbox_mask, image_masked = self.bbox_mask(image, bg)

        # Apply neural network to predict bounding boxes
        bboxes: np.ndarray = self.model_frcnn(image)

        # Filter those which do not fall in the the change map or are too small
        index: t.List[int] = []
        for i, (x, y, x_, y_) in enumerate(bboxes):
            if bbox_mask[int(np.round((y + y_) / 2))][int(np.round((x + x_) / 2))] != 255:
                continue
            if (y_ - y) * (x_ - x) < self.min_bbox_area:
                continue
            index.append(i)
        
        # Nothing was detected, so no lanes
        if len(bboxes) == 0:
            return np.zeros_like(np.array(data['query'])).tolist()

        # Otherwise we get only those in the change map
        bboxes = bboxes[index]

        # Filter out BBOXes not close to a zone
        bboxes = bboxes[(self.distance(bboxes, self.assets.zone_all) < self.max_zone_distance).ravel()]

        # Separate BBOXes into three zones: A, B, C
        bboxes_A: np.ndarray = bboxes[(self.distance(bboxes, self.assets.zone_a) < self.max_zone_distance).ravel()]
        bboxes_B: np.ndarray = bboxes[(self.distance(bboxes, self.assets.zone_b) < self.max_zone_distance).ravel()]
        bboxes_C: np.ndarray = bboxes[(self.distance(bboxes, self.assets.zone_c) < self.max_zone_distance).ravel()]

        # Separate BBOXes into lanes: 0-8
        bboxes_lanes: t.List[np.ndarray] = []
        bboxes_lanes.append(self.distance(bboxes_A, self.assets.zones[0], pts='a'))
        bboxes_lanes.append(self.distance(bboxes_A, self.assets.zones[1], pts='a'))
        bboxes_lanes.append(self.distance(bboxes_A, self.assets.zones[2], pts='a'))
        bboxes_lanes.append(self.distance(bboxes_B, self.assets.zones[3], pts='b'))
        bboxes_lanes.append(self.distance(bboxes_B, self.assets.zones[4], pts='b'))
        bboxes_lanes.append(self.distance(bboxes_B, self.assets.zones[5], pts='b'))
        bboxes_lanes.append(self.distance(bboxes_C, self.assets.zones[6], pts='c'))
        bboxes_lanes.append(self.distance(bboxes_C, self.assets.zones[7], pts='c'))
        bboxes_lanes.append(self.distance(bboxes_C, self.assets.zones[8], pts='c'))

        # Filter out based on the lane points distance
        index_A: np.ndarray = np.min(np.concatenate(bboxes_lanes[ :3], axis=1), axis=1) < self.max_lane_distance
        bboxes_A = bboxes_A[index_A, ...]
        index_B: np.ndarray = np.min(np.concatenate(bboxes_lanes[3:6], axis=1), axis=1) < self.max_lane_distance
        bboxes_B = bboxes_B[index_B, ...]
        index_C: np.ndarray = np.min(np.concatenate(bboxes_lanes[6: ], axis=1), axis=1) < self.max_lane_distance
        bboxes_C = bboxes_C[index_C, ...]

        # Find out the associated lanes for each zone
        lanes_A: np.ndarray = np.argmin(np.concatenate(bboxes_lanes[ :3], axis=1)[index_A, :], axis=1)
        lanes_B: np.ndarray = np.argmin(np.concatenate(bboxes_lanes[3:6], axis=1)[index_B, :], axis=1) + 3
        lanes_C: np.ndarray = np.argmin(np.concatenate(bboxes_lanes[6: ], axis=1)[index_C, :], axis=1) + 6

        # # Draw BBOXes
        # for i, (bboxes_zone, lanes) in enumerate([(bboxes_A, lanes_A), (bboxes_B, lanes_B), (bboxes_C, lanes_C)]):
        #     # Indicate the zone bbox
        #     for j, bbox in enumerate(bboxes_zone):
        #         x, y, x_c, y_c = bbox.round().astype(np.int32).tolist()
        #         image = cv.rectangle(image, (x, y), (x_c, y_c), (255 if i == 0 else 0, 255 if i == 1 else 0, 255 if i == 2 else 0), 5)
        #         lane = lanes[j]
        #         image = cv.putText(image, str(lane + 1), (int(0.5 * bbox[0] + 0.5 * bbox[2]), int(0.5 * bbox[1] + 0.5 * bbox[3])), cv.FONT_HERSHEY_SIMPLEX, 1, color=(255, 255, 255), thickness=3)
        # cv.imshow('Image mask', np.hstack((image, image_masked)))
        # cv.waitKey(0)

        # Get the unique occupied lanes
        answer: t.List[int] = []
        lanes_occupied = (np.unique(np.concatenate([lanes_A, lanes_B, lanes_C])) + 1).tolist()
        for lane in data['query']:
            if lane in lanes_occupied:
                answer.append(1)
            else:
                answer.append(0)
        return answer


class VehicleTracking(object):
    def __init__(self, assets: Assets, device: torch.device, *args, **kwargs) -> None:
        super().__init__()

    def __call__(self, data: CarTrafficTaskTwoDict) -> t.List[int]:
        return [0, 0, 0, 0]