import numpy as np


class NoDominoFoundError(Exception):
    def __init__(self, unit_matrix: np.ndarray, clean_matrix: np.ndarray, pred_matrix: np.ndarray, prev_matrix: np.ndarray) -> None:
        self.unit_matrix = unit_matrix.copy()
        self.clean_matrix = clean_matrix.copy()
        self.pred_matrix = pred_matrix.copy()
        self.prev_matrix = prev_matrix.copy()
        super().__init__('NoDominoFoundError:\nUnit Matrix:\n{}\nClean Matrix:\n{}\nPred Matrix:\n{}\nPrevMatrix:\n{}'.format(unit_matrix, clean_matrix, pred_matrix, prev_matrix))


class CannotSolveGameError(Exception):
    def __init__(self, game_index: int, game_move_index: int) -> None:
        super().__init__('CannotSolveGameError: {}-{}'.format(game_index, game_move_index))

