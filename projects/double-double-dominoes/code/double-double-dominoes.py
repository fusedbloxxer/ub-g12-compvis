import domino.args
import cv2 as cv
import typing as t
import pathlib as pb
import numpy as np
import random as rnd
import matplotlib.pyplot as plt

if __name__ != '__main__':
    exit(0)

# Construct CLI Interface
args = domino.args.get_args()
print(args.path)

# Predefined expected paths
ROOT_PATH = pb.Path('..')
DATA_PATH = ROOT_PATH / 'data'
HELP_PATH = DATA_PATH / 'help'
TEST_PATH = DATA_PATH / 'test'
TRAIN_PATH = DATA_PATH / 'train'
VALID_PATH = DATA_PATH / 'valid'

# Obtain deterministic results
SEED = 42
rnd.seed(SEED)
np.random.seed(SEED)

# Test
down_scale = 0.25
path = TRAIN_PATH / 'input' / 'regular_task' / '5_20.jpg'
img = cv.imread(str(path.absolute()), cv.IMREAD_COLOR)
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img = cv.resize(img, dsize=None, fx=down_scale, fy=down_scale, interpolation=cv.INTER_LINEAR)
cv.imshow('window_name', img)
cv.waitKey(0)
cv.destroyAllWindows()
