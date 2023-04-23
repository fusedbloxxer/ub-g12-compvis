import abc
import typing as t
from typing import Tuple
import numpy as np
import pandas as pd
import pathlib as pb
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame

from .vision import Board2MatrixOpeation, DisplayOperation
from .data import DDDGameDataset, DDDRegularGameDataset, DDDBonusGameDataset
from .model import CellClassifier, PretrainedCellClassifier, RandomForestCellClassifier


TASK_RESULT = t.TypeVar('TASK_RESULT')


class Task(abc.ABC, t.Generic[TASK_RESULT]):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @abc.abstractmethod
    def solve(self, output_path: t.Optional[pb.Path] = None) -> TASK_RESULT:
        raise NotImplementedError()

    @abc.abstractmethod
    def score(self, output: t.Union[pb.Path, TASK_RESULT]) -> TASK_RESULT:
        raise NotImplementedError()


class DoubleDoubleDominoes(Task[t.Tuple[pd.DataFrame, pd.DataFrame]]):
    def __init__(self,
                 dataset_path: pb.Path,
                 classifier_path: pb.Path,
                 *args,
                 template_index: int | None = None,
                 template_selection_type: str = 'pts',
                 template_dynamic_retrieval: bool=False,
                 show_matrix: bool = False,
                 show_image: bool = False,
                 cell_splitter: str='hough',
                 cell_classifier: str = 'resnet',
                 train: bool = True,
                 **kwargs) -> None:
        super().__init__()

        # Use dataset with / without labels
        self.dataset: DDDGameDataset = DDDGameDataset(dataset_path, train=train)

        # Use a pretrained model to classify the cell contents
        if cell_classifier == 'resnet':
            self.cell_classifier: CellClassifier = PretrainedCellClassifier.load_checkpoint(classifier_path)
        elif cell_classifier == 'forest':
            self.cell_classifier: CellClassifier = RandomForestCellClassifier.load_checkpoint(classifier_path)

        # Game templates: board + grid
        self.template_image_board: np.ndarray = self.dataset.dataset_help.template(dynamic_retrieval=template_dynamic_retrieval,
                                                                                   selection_type=template_selection_type,
                                                                                   template_type='board',
                                                                                   index=template_index,
                                                                                   scale=0.35)
        self.template_image_grid: np.ndarray = self.dataset.dataset_help.template(dynamic_retrieval=template_dynamic_retrieval,
                                                                                   selection_type=template_selection_type,
                                                                                   template_type='grid',
                                                                                   index=template_index,
                                                                                   scale=0.35)

        # ComputerVision Useful Operations
        self.board2matrix = Board2MatrixOpeation(board_template=self.template_image_board,
                                                 grid_template=self.template_image_grid,
                                                 cell_classifier=self.cell_classifier,
                                                 cell_splitter=cell_splitter,
                                                 show_matrix=show_matrix,
                                                 show_image=show_image)

        # Create tasks to be solved
        self.task_regular: RegularTask = RegularTask(dataset=self.dataset.dataset_regular, board2matrix=self.board2matrix)
        self.task_bonus: BonusTask = BonusTask(dataset=self.dataset.dataset_bonus, board2matrix=self.board2matrix)

    def solve(self, output_path: t.Optional[pb.Path] = None) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
        # Solve the two tasks and optionally save the results in the their respective format
        regular_solution = self.task_regular.solve(output_path)
        bonus_solution = self.task_bonus.solve(output_path)
        return regular_solution, bonus_solution

    def score(self, output: pb.Path | Tuple[DataFrame, DataFrame]) -> Tuple[DataFrame, DataFrame]:
        if isinstance(output, pb.Path):
            regular_score = self.task_regular.score(output)
            bonus_score = self.task_bonus.score(output)
        else:
            regular_score = self.task_regular.score(output[0])
            bonus_score = self.task_regular.score(output[1])
        return regular_score, bonus_score


class RegularTask(Task[pd.DataFrame]):
    def __init__(self, dataset: DDDRegularGameDataset, board2matrix: Board2MatrixOpeation, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dataset: DDDRegularGameDataset = dataset
        self.board2matrix: Board2MatrixOpeation = board2matrix

    def solve(self, output_path: pb.Path | None = None) -> DataFrame:
        return pd.DataFrame()

    def score(self, output: pb.Path | DataFrame) -> DataFrame:
        return super().score(output)


class BonusTask(Task[pd.DataFrame]):
    def __init__(self, dataset: DDDBonusGameDataset, board2matrix: Board2MatrixOpeation, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dataset: DDDBonusGameDataset = dataset
        self.board2matrix: Board2MatrixOpeation = board2matrix

    def solve(self, output_path: pb.Path | None = None) -> DataFrame:
        return pd.DataFrame()

    def score(self, output: pb.Path | DataFrame) -> DataFrame:
        return super().score(output)


class Game(object):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()


class GameTurn(object):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

