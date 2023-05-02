import cv2 as cv
import typing as t
import pandas as pd
import pathlib as pb
import numpy as np
import sklearn as sk
import random as rnd
import matplotlib.pyplot as plt
import torch
import torchvision as tv
import torch.utils.data as data
import argparse
from torch import Tensor
import lightning.pytorch as pl

import domino.args
import domino.data
import domino.game
import domino.model
import domino.vision
import domino.preprocess
from domino.model import CellClassifier


if __name__ != '__main__':
    exit(0)

# Construct CLI Interface
args: argparse.Namespace = domino.args.get_args()

# Predefined expected paths
ROOT_PATH: pb.Path = pb.Path('.')
HELP_PATH: pb.Path = ROOT_PATH / 'template'
CACHE_DIR: pb.Path = ROOT_PATH / '.cache'
MODEL_DIR: pb.Path = ROOT_PATH / '.model'
SKLRN_DIR: pb.Path = MODEL_DIR / 'sklearn'
TORCH_DIR: pb.Path = MODEL_DIR / 'torch'

# Pretrained model weights
FOREST_MODEL_PATH: pb.Path = SKLRN_DIR / 'balanced-random-forest-full.joblib'
RESNET_MODEL_PATH: pb.Path = TORCH_DIR / 'resnet50-finetune-full.ckpt'
SVM_MODEL_PATH: pb.Path    = SKLRN_DIR / 'svm-full.joblib'

# Obtain deterministic results
SEED = 42
rnd.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
generator: torch.Generator = torch.Generator().manual_seed(SEED)

# Configure current environment
torch.hub.set_dir(CACHE_DIR)
torch.set_float32_matmul_precision('medium')

# Construct the possible classifiers (as a fallback measure)
classifiers: t.List[CellClassifier] = [
    CellClassifier.load_checkpoint(RESNET_MODEL_PATH, 'resnet'),
    CellClassifier.load_checkpoint(SVM_MODEL_PATH, 'svm'),
    CellClassifier.load_checkpoint(FOREST_MODEL_PATH, 'forest'),
]

# Construct task solver
task_solver = domino.game.DoubleDoubleDominoesTask(
    dataset_path=args.input,
    template_path=HELP_PATH,
    output_path=args.output,
    cell_splitter=args.grid,
    classifiers=classifiers,
    show_matrix=args.show_matrix,
    show_image=args.show_image,
    debug=args.debug,
    train=args.train,
)

# Solve the task (hopefully)
task_solver.solve()
