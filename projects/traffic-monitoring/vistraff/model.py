import typing as t
import torch
import numpy as np
from ultralytics import YOLO
from torchvision.transforms.functional import to_pil_image 
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

from res import Assets


class FasterRCNNMOD():
    def __init__(self, device: torch.device, box_score_thresh=0.7, box_nms_thresh=0.7) -> None:
        super().__init__()
        self.classes: t.List[str] = ['truck', 'bus', 'motorcycle', 'car']
        self.weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn_v2(weights=self.weights,
                                                            box_score_thresh=box_score_thresh,
                                                            box_nms_thresh=box_nms_thresh).to(device)
        self.preprocess = self.weights.transforms()
        self.model.eval()
        self.device = device

    def __call__(self, image: np.ndarray, with_scores: bool=False) -> np.ndarray | torch.Tensor:
        image = to_pil_image(image)
        pred = self.model([self.preprocess(image).to(self.device)])
        boxes, labels, scores, clss = [], [], [], []

        for i in range(len(pred[0]['labels'])):
            if self.weights.meta['categories'][pred[0]['labels'][i]] not in self.classes:
                continue
            clss.append(pred[0]['labels'][i])
            boxes.append(pred[0]['boxes'][i])
            scores.append(pred[0]['scores'][i])
            labels.append(self.weights.meta['categories'][pred[0]['labels'][i]])

        if with_scores is False:
            if len(boxes) != 0:
                return torch.stack(boxes, dim=0).cpu().numpy()
            return np.array([])

        if len(boxes) != 0:
            return torch.cat([
                torch.stack(boxes, dim=0),
                torch.stack(scores)[..., None],
                torch.stack(clss)[..., None]
            ], dim=1).cpu()
        return torch.empty((0, 5))


class YOLOMOD():
    def __init__(self, assets: t.Optional[Assets] = None) -> None:
        super().__init__()
        self.classes: t.List[str] = ['truck', 'bus', 'motorcycle', 'car']
        self.assets: Assets | None = assets

        if assets is None:
            self.model = YOLO('yolov8n.pt', task='detect')
        elif assets is not None:
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
