import typing as t
import pathlib as pb
import torch
from torchvision.ops import box_iou
import cv2 as cv
import numpy as np
from ultralytics import YOLO
from ocsort.ocsort import OCSort

from res import Assets
from model import FasterRCNNMOD
from data import CarTrafficTaskOneDict, CarTrafficTaskTwoDict


class VehicleOccupancy(object):
    def __init__(self, assets: Assets, device: torch.device, *args, debug: bool=False, **kwargs) -> None:
        super().__init__()
        self.device = device
        self.assets: Assets = assets
        self.debug = debug

        # Initialize models
        self.model_frcnn = FasterRCNNMOD(device=device, box_score_thresh=0.25, box_nms_thresh=0.35)

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

        if self.debug:
            # Draw BBOXes
            for i, (bboxes_zone, lanes) in enumerate([(bboxes_A, lanes_A), (bboxes_B, lanes_B), (bboxes_C, lanes_C)]):
                # Indicate the zone bbox
                for j, bbox in enumerate(bboxes_zone):
                    x, y, x_c, y_c = bbox.round().astype(np.int32).tolist()
                    image = cv.rectangle(image, (x, y), (x_c, y_c), (255 if i == 0 else 0, 255 if i == 1 else 0, 255 if i == 2 else 0), 5)
                    lane = lanes[j]
                    image = cv.putText(image, str(lane + 1), (int(0.5 * bbox[0] + 0.5 * bbox[2]), int(0.5 * bbox[1] + 0.5 * bbox[3])), cv.FONT_HERSHEY_SIMPLEX, 1, color=(255, 255, 255), thickness=3)
            cv.imshow('Image mask', np.hstack((image, image_masked)))
            cv.waitKey(0)

        # Get the unique occupied lanes
        answer: t.List[int] = []
        lanes_occupied = (np.unique(np.concatenate([lanes_A, lanes_B, lanes_C])) + 1).tolist()
        for lane in data['query']:
            if lane in lanes_occupied:
                answer.append(1)
            else:
                answer.append(0)
        return answer


class VehicleTracks(object):
    def __init__(self, method: str, *args, **kwargs) -> None:
        super().__init__()
        self.method = method

        self.idx = 0
        self.frame = 0
        self.path: t.List[torch.Tensor] = []
        self.ord: t.Dict[int, int] = {}
        self.bbox: t.Dict[int, t.List[torch.Tensor]] = {}

    @property
    def last_idx(self) -> int:
        return self.idx

    @last_idx.setter
    def last_idx(self, idx: int) -> None:
        if idx not in self.ord:
            self.ord[idx] = len(self.ord) + 1
        self.idx = idx

    @property
    def last_bbox(self) -> torch.Tensor:
        return self.path[-1]

    @last_bbox.setter
    def last_bbox(self, bbox: torch.Tensor) -> None:
        if self.last_idx not in self.bbox:
            self.bbox[self.last_idx] = [bbox]
        else:
            self.bbox[self.last_idx].append(bbox)
        self.path.append(bbox)

    def bbox_tuple(self) -> t.Tuple[int, int, int, int]:
        return tuple(self.last_bbox.type(torch.int32).tolist())


class VehicleTracking(object):
    def __init__(self, assets: Assets, device: torch.device, debug: bool = False) -> None:
        super().__init__()

        # Configuration
        self.debug = debug
        self.device = device
        self.assets: Assets = assets
        self.debug_image: np.ndarray = np.empty((self.assets.image_height, self.assets.image_width))

        # Predictors
        self.classes: t.List[int] = [2, 3, 5, 7]
        self.detector_yolo = YOLO(model=str(self.assets.weights_yolo), task='detect')
        self.detector_frcnn = FasterRCNNMOD(device=device, box_score_thresh=0.25, box_nms_thresh=0.10)

        # Trackers
        self.frame = -1
        self.tracker = OCSort()
        self.max_frames_lost = 35 # TODO! More, less?
        self.tracks = VehicleTracks(method='yolo_bytetrack')

    def __call__(self, data: CarTrafficTaskTwoDict) -> t.Tuple[int, int, int, int] | None:
        try:
            # Update current index
            self.frame += 1
            image: np.ndarray = data['image']
            self.debug_image = data['image'].copy()

            # Obtain detections using YOLO -> FasterRCNN
            bboxes, ids = self.track(image, ['yolo', 'frcnn'])
            assert bboxes is not None and ids is not None, 'No detections were found at all!'
            self.__debug_draw_bboxes(bboxes, ids)

            # If it's the current frame then obtain the id that overlaps the most the ground truth
            if self.frame == 0:
                return self.init(data, bboxes, ids)

            # An ID was found. It is more likely to be what we want because of OCSORT ID switching
            if 0 != len(seen := set(ids).intersection(set(self.tracks.bbox.keys()))):
                return self.found_id(seen, bboxes, ids)

            # Obtain detections using FasterRCNN
            bboxes, ids = self.track(image, ['frcnn'])
            assert bboxes is not None and ids is not None, 'No detections were found at all!'
            self.__debug_draw_bboxes(bboxes, ids)

            # An ID was found. It is more likely to be what we want because of OCSORT ID switching
            if 0 != len(seen := set(ids).intersection(set(self.tracks.bbox.keys()))):
                return self.found_id(seen, bboxes, ids)

            # The entity might change across the video
            return self.missing_id(image, bboxes, ids)
        except Exception as _:
            self.tracks.frame = self.frame
            if len(self.tracks.path) != 0:
                return self.tracks.bbox_tuple()
            return 0, 0, 0, 0

    def init(self, data: CarTrafficTaskTwoDict, bboxes: torch.Tensor, ids: t.List[int]) -> t.Tuple[int, int, int, int]:                
        # Retrieve initial ground truth
        bbox_gt: torch.Tensor = data['bbox']

        # See what's the best overlap
        iou: torch.Tensor = box_iou(bbox_gt.unsqueeze(0), bboxes)
        ind: int = torch.argmax(iou).type(torch.int32).item()
        idx: int = ids[ind]

        # Mark down what is the initial id
        self.tracks.last_idx = idx
        self.tracks.last_bbox = bbox_gt
        return self.tracks.bbox_tuple()

    def found_id(self, seen: t.Set[int], bboxes: torch.Tensor, ids: t.List[int]) -> t.Tuple[int, int, int, int]:
        # Pick the earliest ID (would have more priority)
        early_ids: int = sorted([(self.tracks.ord[i], i) for i in seen], key=lambda x: x[0])

        # Change last ID and BBOX
        self.tracks.frame = self.frame
        self.tracks.last_idx = early_ids[0][1]
        self.tracks.last_bbox = bboxes[ids.index(self.tracks.last_idx)]
        return self.tracks.bbox_tuple()

    def missing_id(self, image: np.ndarray, bboxes: torch.Tensor, ids: t.List[int]) -> t.Tuple[int, int, int, int] | None:
        # If the object is missing for a significant amount of frames stop
        if self.frame - self.tracks.frame > self.max_frames_lost:
            return None

        # Choose BBOX in case of success
        def match_found(idx: int):
            self.tracks.last_idx = idx
            self.tracks.frame = self.frame
            self.tracks.last_bbox = bboxes[index]
            return self.tracks.bbox_tuple()

        # Choose the previous BBOX in the case of failure
        def no_match_found():
            return self.tracks.bbox_tuple()

        # Try to find the best matching box with the previous one - IoU
        iou: torch.Tensor = box_iou(self.tracks.last_bbox.unsqueeze(0), bboxes)
        index: int = torch.argmax(iou).type(torch.int32).item()
        max_iou: torch.Tensor = torch.max(iou)
        idx: int = ids[index]

        # Consider a high margin for IOU to eliminate false positives
        if max_iou >= 0.5:
            return match_found(idx)

        # Compute the centroids
        all_centroids = centroid(bboxes)
        prev_centroid = centroid(self.tracks.last_bbox)
        self.__debug_draw_centroids(prev_centroid, all_centroids)

        # Try to find the best matching box with the previous one - Euclidean
        dist = torch.norm(prev_centroid.unsqueeze(0) - all_centroids, p=2, dim=1)
        values, indices = torch.sort(dist, dim=0)

        # In the case of a single match apply a small threshold:
        if indices.shape[0] == 1:
            bbox: torch.Tensor = self.tracks.last_bbox
            bbox_diag: torch.Tensor = torch.tensor(((bbox[0] - bbox[2]), (bbox[3] - bbox[1])))
            bbox_diag = torch.norm(bbox_diag, p=2, dim=0)

            # A small car should have another small bbox around it!
            if values[0] < bbox_diag * 1.25:
                return match_found(indices[0])
            return no_match_found()

        # For more matches apply Lowe's ratio test + a similar threshold as above:
        if values[0] / values[1] < 0.5:
            bbox: torch.Tensor = self.tracks.last_bbox
            bbox_diag: torch.Tensor = torch.tensor(((bbox[0] - bbox[2]), (bbox[3] - bbox[1])))
            bbox_diag = torch.norm(bbox_diag, p=2, dim=0)

            # A small car should have another small bbox around it!
            if values[indices[0]] < bbox_diag * 1.25:
                return match_found(indices[0])

        # Otherwise we have no option left and will wait for another possible match
        return no_match_found()

    def track(self, image: np.ndarray, detectors: t.List[str]) -> t.Tuple[torch.Tensor, t.List[int]] | t.Tuple[None, None]:
        detections: torch.Tensor = torch.empty(0)

        # Try all specified detectors in order until one succeeds
        for detector in detectors:
            if (detections := self.detect(image, detector)).numel() != 0:
                break

        # Still no result
        if detections.numel() == 0:
            return None, None

        # Fetch tracked entities along with their ids: [bboxes, ids, classes, scores]
        targets: np.ndarray = self.tracker.update(detections, None)

        # Still no result
        if targets.size == 0:
            return None, None

        # Extract BBOXes and Entity IDs
        ids: t.List[int] = torch.tensor(targets[:, 4], dtype=torch.int32).tolist()
        bboxes: torch.Tensor = torch.tensor(targets[:, :4])
        return bboxes, ids

    def detect(self, image: np.ndarray, detector: str) -> torch.Tensor:
        if detector == 'yolo':
            result: torch.Tensor = self.detector_yolo(image, device=self.device, classes=self.classes)[0]
            detections = torch.cat([result.boxes.xyxy, result.boxes.conf[..., None], result.boxes.cls[..., None]], dim=1).cpu()
            return detections
        elif detector == 'frcnn':
            detections = self.detector_frcnn(image, with_scores=True)
            return detections
        else:
            raise NotImplementedError(f'Detector does not exist: {detector}')

    def __debug_draw_centroids(self, centroid: np.ndarray, centroids: np.ndarray) -> None:
        if not self.debug:
            return None
        return None

    def __debug_draw_bboxes(self, bboxes: torch.Tensor, ids: t.List[int]) -> None:
        if not self.debug:
            return None

        # Debug results with their IDs
        for i, bbox in enumerate(bboxes):
            # BBOX + ID
            track_id = ids[i]
            bbox = bbox.type(torch.int32).tolist()
            x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
            x_c, y_c = int(0.5 * bbox[0] + 0.5 *bbox[2]), int(0.5 * bbox[1] + 0.5 * bbox[3])

            # Draw BBOX + ID
            self.debug_image = cv.rectangle(self.debug_image, (x, y, w, h), color=(0, 0, 255), thickness=5)
            self.debug_image = cv.putText(self.debug_image, f'ID: {track_id}', (x_c, y_c), cv.FONT_HERSHEY_SIMPLEX, 2, (0,) * 3, 3)
            cv.imshow('DEBUG', self.debug_image)


def centroid(bbox: torch.Tensor) -> torch.Tensor:
    if bbox.ndim == 1:
        return torch.tensor([0.5 * bbox[0] + 0.5 * bbox[2], 0.5 * bbox[1] + 0.5 * bbox[3]])
    else:
        return torch.column_stack([0.5 * bbox[:, 0] + 0.5 * bbox[:, 2], 0.5 * bbox[:, 1] + 0.5 * bbox[:, 3]])
