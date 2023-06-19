import abc
from abc import ABC
import torch
import numpy as np
import typing as t
import pathlib as pb
from typing import Any
from ultralytics import YOLO
from torchvision.transforms.functional import to_pil_image 
from torchvision.models.detection import ssd300_vgg16, fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection import SSD300_VGG16_Weights, FasterRCNN_ResNet50_FPN_V2_Weights

from .res import Assets


class FasterRCNNMOD():
    def __init__(self, *args, box_score_thresh=0.7, box_nms_thresh=0.7, **kwargs) -> None:
        super().__init__()
        self.classes: t.List[str] = ['truck', 'bus', 'motorcycle', 'car']
        self.weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn_v2(weights=self.weights, box_score_thresh=box_score_thresh, box_nms_thresh=box_nms_thresh)
        self.preprocess = self.weights.transforms()
        self.model.eval()

    def __call__(self, image: np.ndarray) -> np.ndarray:
        image = to_pil_image(image)
        pred = self.model([self.preprocess(image)])
        boxes, labels = [], []

        for i in range(len(pred[0]['labels'])):
            if self.weights.meta['categories'][pred[0]['labels'][i]] not in self.classes:
                continue
            boxes.append(pred[0]['boxes'][i])
            labels.append(self.weights.meta['categories'][pred[0]['labels'][i]])

        if len(boxes) != 0:
            return torch.stack(boxes, dim=0).cpu().numpy()
        else:
            return np.array([])


class YOLOMOD():
    def __init__(self, *args, assets: t.Optional[Assets] = None, **kwargs) -> None:
        super().__init__()
        self.classes: t.List[str] = ['truck', 'bus', 'motorcycle', 'car']
        self.assets = assets

        if assets is None:
            self.model = YOLO('yolov8n.pt', task='detect')
        else:
            self.model = YOLO(model=self.assets.weights_path / 'yolo_uadetrac.pt', task='detect')

    def __call__(self, image: np.ndarray) -> np.ndarray:
        result = self.model([image])[0]
        bboxes: torch.Tensor = result.boxes.xyxy
        labels = [result.names[c] for c in result.boxes.cls.tolist()]

        index: t.List[int] = []
        for i, label in enumerate(labels):
            if label not in self.classes:
                continue
            index.append(i)

        return bboxes[index].cpu().numpy()
