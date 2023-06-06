import torch
import torchvision as tv
# import random
import cv2 as cv
# import numpy as np
# import typing as t
# import pathlib as pb
# from ultralytics import YOLO
# from strongsort.strong_sort import StrongSORT

# import vistraff as vt
# from vistraff.data import CarTrafficDataset, CarTrafficVideo


# PATH_ROOT: pb.Path = pb.Path('.')
# PATH_WEIGHTS: pb.Path = PATH_ROOT / 'weights'
# PATH_DATA: pb.Path = PATH_ROOT / 'data' / 'train'
# SEED: int = 787

# np.random.default_rng(SEED)
# torch.manual_seed(SEED)
# np.random.seed(SEED)
# random.seed(SEED)

# dataset = CarTrafficDataset(PATH_DATA)
i = cv.imread('data/train/Task1/01_1.jpg')
cv.imwrite('./test_image.jpg', i)
cv.imshow('raw image', i)
cv.waitKey(0)
cv.destroyAllWindows()