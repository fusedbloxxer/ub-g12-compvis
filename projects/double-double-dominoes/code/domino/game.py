import abc
import cv2 as cv
import typing as t
from typing import Tuple
import numpy as np
import pandas as pd
import os
import pathlib as pb
import copy
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame

from .errors import NoDominoFoundError, CannotSolveGameError
from .vision import Board2MatrixOpeation, DisplayOperation
from .data import Dataset, GameDataset, DDDGameDataset, DDDRegularGameDataset, DDDBonusGameDataset
from .model import CellClassifier, PretrainedCellClassifier, RandomForestCellClassifier


TASK_DATASET = t.TypeVar('TASK_DATASET', bound=Dataset)


class Task(abc.ABC, t.Generic[TASK_DATASET]):
    def __init__(self, game_dataset: TASK_DATASET, *args, **kwargs) -> None:
        super().__init__()
        self.dataset: TASK_DATASET = game_dataset

    @abc.abstractmethod
    def solve(self) -> None:
        raise NotImplementedError()

    @property
    def train(self) -> bool:
        if isinstance(self.dataset, (GameDataset, DDDGameDataset)):
            return self.dataset.train
        else:
            return False


class DoubleDoubleDominoesTask(Task[DDDGameDataset]):
    def __init__(self,
                 dataset_path: pb.Path,
                 template_path: pb.Path,
                 output_path: pb.Path,
                 classifiers: t.List[CellClassifier],
                 *args,
                 template_index: t.Optional[int] = None,
                 template_selection_type: str = 'pts',
                 template_dynamic_retrieval: bool=False,
                 show_matrix: bool = False,
                 show_image: bool = False,
                 cell_splitter: str='hough',
                 train: bool = False,
                 debug: bool = False,
                 **kwargs) -> None:
        super().__init__(*args, game_dataset=DDDGameDataset(dataset_path, path_template=template_path, train=train, path_output=output_path), **kwargs)

        # Use a pretrained model to classify the cell contents
        self.debug: bool = debug
        self.classifiers: t.List[CellClassifier] = classifiers
        assert len(classifiers) >= 1, 'At least on cell classifier must be provided!'

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
                                                 cell_classifier=self.classifiers[0],
                                                 cell_splitter=cell_splitter,
                                                 show_matrix=show_matrix,
                                                 show_image=show_image)

        # Create tasks to be solved
        self.task_regular: RegularTask = RegularTask(context=self,
                                                     dataset=self.dataset.dataset_regular,
                                                     board2matrix=self.board2matrix,
                                                     debug=debug)
        self.task_bonus: BonusTask = BonusTask(context=self,
                                               dataset=self.dataset.dataset_bonus,
                                               board2matrix=self.board2matrix,
                                               debug=debug)

    def solve(self) -> None:
        # Solve the two tasks and optionally save the results in the their respective format
        self.task_regular.solve()
        self.task_bonus.solve()


class RegularTask(Task[DDDRegularGameDataset]):
    def __init__(self,
                 context: DoubleDoubleDominoesTask,
                 dataset: DDDRegularGameDataset,
                 board2matrix: Board2MatrixOpeation, *args, debug: bool = False, **kwargs) -> None:
        super().__init__(*args, game_dataset=dataset, **kwargs)
        self.board2matrix: Board2MatrixOpeation = board2matrix
        self.context: DoubleDoubleDominoesTask = context
        self.debug: bool = debug

    def solve(self) -> None:
        for game_index in range(len(self.dataset)):
            # Read one game at a time
            game_images, game_moves = self.dataset[game_index]

            try:
                # Build a single instance of the game to isolate states
                game = Game(task=self,
                            index=game_index + 1,
                            images=game_images,
                            moves=game_moves,
                            board2matrix=self.board2matrix,
                            debug=self.debug)

                # Solve the game
                game.solve(self.dataset.path_output)
            except CannotSolveGameError as e:
                if self.debug:
                    print('Skipping Game: {}'.format(game_index))
                    print(e)


class BonusTask(Task[DDDBonusGameDataset]):
    def __init__(self,
                 context: DoubleDoubleDominoesTask,
                 dataset: DDDBonusGameDataset,
                 board2matrix: Board2MatrixOpeation, *args, debug: bool = False, **kwargs) -> None:
        super().__init__(*args, game_dataset=dataset, **kwargs)
        self.board2matrix: Board2MatrixOpeation = board2matrix
        self.context: DoubleDoubleDominoesTask = context
        self.debug: bool = debug

    def solve(self) -> None:
        pass


class Game(object):
    def __init__(self,
                 task: RegularTask,
                 index: int,
                 images: np.ndarray,
                 moves: pd.DataFrame,
                 board2matrix: Board2MatrixOpeation, *args, debug: bool = False, **kwargs) -> None:
        super().__init__()

        # Game specific content
        self.b2m: Board2MatrixOpeation = board2matrix
        self.player_moves: pd.DataFrame = moves
        self.game_states: t.List[GameState] = []
        self.game_moves: t.List[GameMove] = []
        self.images: np.ndarray = images
        self.task: RegularTask = task
        self.game_index: int = index
        self.debug: bool = debug

    def solve(self, output: pb.Path) -> None:
        # Initialize the GameState from scratch as it's the first turn
        self.game_states: t.List[GameState] = [GameState.empty(debug=self.debug)]

        try:
            # Iterate through each image and the respective player move
            for move_index, (image, (_, move)) in enumerate(zip(self.images, self.player_moves.iterrows())):
                # Allocate a number of retries for each move
                is_done: bool = False
                retry_count: int = 0

                # Use the best classifier initially
                self.b2m.cell_classifier = self.task.context.classifiers[0]

                # Try to solve the game multiple times
                while not is_done:
                    try:
                        # Image -> Matrix with possible bad predictions
                        matrix: np.ndarray = self.b2m(image)

                        # Generate the next state of the game
                        next_game_state: GameState = self.game_states[-1] + (matrix, move)
                    except NoDominoFoundError as e:
                        if self.debug:
                            print(e)
                        if retry_count == len(self.task.context.classifiers) - 1:
                            raise CannotSolveGameError(self.game_index, move_index)
                        else:
                            self.b2m.cell_classifier = self.task.context.classifiers[retry_count := retry_count + 1]
                        if self.debug:
                            print(f'Retry {retry_count}...')
                    except Exception as e:
                        if self.debug:
                            print('Unknown Error:\n{}', e)
                            raise CannotSolveGameError(self.game_index, move_index)
                    else:
                        is_done = True

                # Go to the next state and retrieve the next move
                self.game_states.append(next_game_state)
        except CannotSolveGameError as e:
            if self.debug:
                print('Saving results obtained until this point for the current game: {}'.format(self.game_index))

        # Extract the list of moves that have occurred
        self.game_moves = [state.game_move for state in self.game_states if state.game_move is not None]

        # Save the results according to the established format as .txt files
        for i, move in enumerate(self.game_moves):
            # Create output directory if it does not exist
            output.mkdir(parents=True, exist_ok=True)

            # Write the solution
            with open(str(output / f'{self.game_index}_{i + 1:02}.txt'), 'w') as move_file:
                move_file.write(str(move))
                move_file.flush()


class GameState(object): # TODO: remove debug
    def __init__(self, move_index: int, game_board: 'GameBoard', score_track: 'GameScoreTrack', *args, game_move: t.Optional['GameMove'] = None, debug: bool = False, **kwargs) -> None:
        super().__init__()
        self.debug: bool = debug
        self.move_index: int = move_index
        self.game_board: GameBoard = game_board
        self.score_track: GameScoreTrack = score_track
        self.game_move: t.Optional[GameMove] = game_move

    def __add__(self, change: t.Tuple[np.ndarray, pd.Series]) -> 'GameState':
        # Unpack the change to extract the predicted cell values and leverage the player turn
        matrix, player_move = change

        # Start robust filtering of invalid predictions
        domino, pos = self.matrix2move(matrix)

        # Find out who the main and other player are
        main_player: str  = 'player' + str(player_move.loc['player'])
        other_player: str = str(GameState.other_player(main_player))

        # Compute the score using the rules of the game
        scores: Tuple[int, int] = self.move2scores(domino, pos, main_player, other_player)

        # Create the game move using the computed score and placement
        game_move: GameMove = GameMove(main_score=scores[0], main_player=main_player, other_score=scores[1], other_player=other_player, domino=domino, positions=pos)
        new_score_track: GameScoreTrack = self.score_track + game_move
        new_board: GameBoard = self.game_board + game_move

        # Create a new game state using immutability
        return GameState(move_index=self.move_index + 1, game_board=new_board, score_track=new_score_track, game_move=game_move, debug=self.debug)

    def matrix2move(self, matrix: np.ndarray) -> t.Tuple['GameDomino', t.Tuple['GamePosition', 'GamePosition']]:
        # Only changes should be taken into account, overlaps should be removed along with isolated unit cells
        unit_matrix: np.ndarray = (matrix != self.game_board.value()).astype(np.int32)
        unit_matrix_init: np.ndarray = unit_matrix.copy()

        if self.debug:
            print('Initial Unit Matrix:\n', unit_matrix)
            print('Prediction:\n', matrix)
            print('Previous Board Values:\n', self.game_board.value())

        unit_matrix = self.remove_invalid_cells(matrix, unit_matrix)
        unit_matrix = self.remove_isolated_cells(unit_matrix)

        # Initial prior that which states that pieces must be centered
        if self.move_index == 0:
            unit_matrix: np.ndarray = self.remove_non_centered_cells(unit_matrix)

        # Detect the possible domino pieces
        dominoes: t.List[t.Tuple[GameDomino, t.Tuple[GamePosition, GamePosition]]] = self.detect_dominoes(matrix, unit_matrix)

        # Remove invalid dominoes by using prior information
        if self.move_index != 0:
            dominoes = self.remove_seen_dominoes(dominoes)
            dominoes = self.remove_invalid_dominoes(dominoes)

        if self.debug:
            print('Filtered Unit Matrix:\n', unit_matrix)
            print('Found Dominoes:\n', [str(piece[0]) + ' at ' + str(piece[1][0]) + ' ' + str(piece[1][1]) for piece in dominoes])

        if len(dominoes) == 0:
            raise NoDominoFoundError(unit_matrix_init, unit_matrix, matrix, self.game_board.value())

        # assert len(dominoes) == 1, f'A single domino may be predicted! Found: {len(dominoes)}'  # TODO: remove this
        return dominoes[0]

    def detect_dominoes(self, matrix: np.ndarray, unit_matrix: np.ndarray) -> t.List[t.Tuple['GameDomino', t.Tuple['GamePosition', 'GamePosition']]]:
        # Keep only dominoes that can exist to minimize invalid moves
        dominoes: t.List[t.Tuple['GameDomino', t.Tuple['GamePosition', 'GamePosition']]] = []

        # Compute the list of all possible dominoes (valid & invalid)
        pos: t.List[t.Tuple[GamePosition, GamePosition]] = []
        for i in range(unit_matrix.shape[0]):
            for j in range(unit_matrix.shape[1]):
                if unit_matrix[i][j] == 0:
                    continue
                if j < unit_matrix.shape[1] - 1 and unit_matrix[i][j + 1] == 1:
                    pos.append((GamePosition(index=(i, j)), GamePosition(index=(i, j + 1))))
                if i < unit_matrix.shape[0] - 1 and unit_matrix[i + 1][j] == 1:
                    pos.append((GamePosition(index=(i, j)), GamePosition(index=(i + 1, j))))

        # Build the domino
        for i, ((i1, j1), (i2, j2)) in enumerate(pos):
            if matrix[i1][j1] >= matrix[i2][j2]:
                domino: GameDomino = GameDomino((matrix[i1][j1], matrix[i2][j2]))
                head_pos: GamePosition = pos[i][0]
                tail_pos: GamePosition = pos[i][1]
            else:
                domino: GameDomino = GameDomino((matrix[i2][j2], matrix[i1][j1]))
                head_pos: GamePosition = pos[i][1]
                tail_pos: GamePosition = pos[i][0]
            dominoes.append((domino, (head_pos, tail_pos)))

        return dominoes

    def remove_invalid_dominoes(self, dominoes: t.List[t.Tuple['GameDomino', t.Tuple['GamePosition', 'GamePosition']]]) -> t.List[t.Tuple['GameDomino', t.Tuple['GamePosition', 'GamePosition']]]:
        valid_dominoes: t.List[t.Tuple['GameDomino', t.Tuple['GamePosition', 'GamePosition']]] = []
        for domino, placement in dominoes:
            # Fetch head's and tail's adjacent cells
            HA, TA = GameBoard.neighbors(placement)

            # Separatte the neighbours from the adjacent cells
            HN = [N for i, N in enumerate(HA) if i % 2 == 0]
            TN = [N for i, N in enumerate(TA) if i % 2 == 0]

            # A piece must be connected to another with the same number
            is_connected: bool = False
            head, tail = domino.head, domino.tail
            for value, neighbors in [(head, HN), (tail, TN)]:
                for neighbor in neighbors:
                    if neighbor is None:
                        continue
                    if (piece := self.game_board[neighbor[0]][neighbor[1]].domino()) is None:
                        continue
                    if value == piece:
                        is_connected = True
                        break
                if is_connected:
                    break
            if not is_connected:
                continue

            # Compute a boolean array indicating if an adjacent cell has a domino on it or not. Outside cells are None.
            HNA = [None if p is None else self.game_board[p[0]][p[1]].domino() for p in HA]
            TNA = [None if p is None else self.game_board[p[0]][p[1]].domino() for p in TA]

            # A piece must not form a square on the corners
            is_square: bool = False
            for CA in [HNA, TNA]:
                if all([x is not None for x in CA[:3]]):
                    is_square = True
                    break
                if all([x is not None for x in CA[-3:]]):
                    is_square = True
                    break
            if HNA[0] is not None and TNA[-1] is not None:
                is_square = True
            if HNA[-1] is not None and TNA[0] is not None:
                is_square = True
            if is_square:
                continue

            # Keep the domino because it might be good
            valid_dominoes.append((domino, placement))
        return valid_dominoes

    def remove_seen_dominoes(self, dominoes: t.List[t.Tuple['GameDomino', t.Tuple['GamePosition', 'GamePosition']]]) -> t.List[t.Tuple['GameDomino', t.Tuple['GamePosition', 'GamePosition']]]:
        return [(domino, pos) for domino, pos in dominoes if domino in self.game_board.dominoes_off]

    def remove_non_centered_cells(self, unit_matrix: np.ndarray) -> np.ndarray:
        unit_matrix = unit_matrix.copy() # TODO: what if there are multiple choices?
        for i in range(unit_matrix.shape[0]):
            for j in range(unit_matrix.shape[1]):
                if unit_matrix[i][j] != 1:
                    continue
                if (i, j) in {(6, 7), (7, 8), (8, 7), (7, 6), (7, 7)}:
                    continue
                unit_matrix[i][j] = 0
        return unit_matrix

    def remove_isolated_cells(self, unit_matrix: np.ndarray) -> np.ndarray:
        unit_matrix = unit_matrix.copy()
        for i in range(unit_matrix.shape[0]):
            for j in range(unit_matrix.shape[1]):
                if unit_matrix[i][j] != 1:
                    continue
                if i > 0 and unit_matrix[i - 1][j] == 1:
                    continue
                if i < unit_matrix.shape[0] - 1 and unit_matrix[i + 1][j] == 1:
                    continue
                if j > 0 and unit_matrix[i][j - 1] == 1:
                    continue
                if j < unit_matrix.shape[1] - 1 and unit_matrix[i][j + 1] == 1:
                    continue
                unit_matrix[i][j] = 0
        return unit_matrix

    def remove_invalid_cells(self, matrix: np.ndarray, unit_matrix: np.ndarray) -> np.ndarray:
        unit_matrix = unit_matrix.copy()
        for i in range(unit_matrix.shape[0]):
            for j in range(unit_matrix.shape[1]):
                if unit_matrix[i][j] != 1:
                    continue
                if matrix[i][j] == 7:
                    matrix[i][j] = 0
                    continue
                if not self.game_board[i][j].is_empty():
                    matrix[i][j] = 0
                    continue
        return unit_matrix

    def move2scores(self, domino: 'GameDomino', pos: t.Tuple['GamePosition', 'GamePosition'], main_player: str, other_player: str) -> t.Tuple[int, int]:
        # Both players have zero for the current move at the beginning
        main_score:  int = 0
        other_score: int = 0

        # Apply the first rule
        if   (diamond := self.game_board[pos[0][0]][pos[0][1]].diamond()) is not None:
            main_score += diamond * (2 if domino.is_double_domino() else 1)
        elif (diamond := self.game_board[pos[1][0]][pos[1][1]].diamond()) is not None:
            main_score += diamond * (2 if domino.is_double_domino() else 1)

        # Apply the second rule
        if self.score_track[main_player] in (domino.head, domino.tail):
            main_score += 3
        if self.score_track[other_player] in (domino.head, domino.tail):
            other_score += 3

        # Return both scores
        return main_score, other_score

    @staticmethod
    def other_player(player: str) -> str:
        if player == 'player1':
            return 'player2'
        elif player == 'player2':
            return 'player1'
        else:
            raise ValueError('Invalid player number: {}'.format(player))

    @staticmethod
    def empty(**kwargs) -> 'GameState':
        # Initialize the first game state (beginning of the game)
        game_board: GameBoard = GameBoard.empty()
        score_track: GameScoreTrack = GameScoreTrack.empty()
        return GameState(move_index=0, game_board=game_board, score_track=score_track, game_move=None, **kwargs)


class GameScoreTrack(object):
    score_track: t.List[t.Optional[int]] = [
        None,
        1, 2, 3, 4, 5, 6, 0, 2, 5, 3, 4, 6, 2, 2, 0, 3, 5, 4, 1, 6,
        2, 4, 5, 5, 0, 6, 3, 4, 2, 0, 1, 5, 1, 3, 4, 4, 4, 5, 0, 6,
        3, 5, 4, 1, 3, 2, 0, 0, 1, 1, 2, 3, 6, 3, 5, 2, 1, 0, 6, 6,
        5, 2, 1, 2, 5, 0, 3, 3, 5, 0, 6, 1, 4, 0, 6, 3, 5, 1, 4, 2,
        6, 2, 3, 1, 6, 5, 6, 2, 0, 4, 0, 1, 6, 4, 4, 1, 6, 6, 3, 0,
    ]

    def __init__(self, player_positions: t.Dict[str, int], *args, **kwargs) -> None:
        super().__init__()
        self.player_positions: t.Dict[str, int] = player_positions

    def __add__(self, move: 'GameMove') -> 'GameScoreTrack':
        # Copy the old state and add over it (immutability)
        new_player_positions: t.Dict[str, int] = copy.deepcopy(self.player_positions)

        # Add the score to both players
        new_player_positions[move.main_player]  += move.main_score
        new_player_positions[move.other_player] += move.other_score

        # Create a new GameScoreTrack to use the updated player positions due to score changes
        return GameScoreTrack(player_positions=new_player_positions)

    def __getitem__(self, player: str) -> t.Optional[int]:
        return GameScoreTrack.score_track[self.player_positions[player]]

    @staticmethod
    def empty() -> 'GameScoreTrack':
        # Both players start from the None position (outside)
        player_positions = dict({
            'player1': 0,
            'player2': 0,
        })

        # Initialize game score starting at the beginning
        return GameScoreTrack(player_positions)


class GameDomino(object):
    dominoes: t.Set[t.Tuple[int, int]] = {
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0),
        (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1),
        (2, 2), (3, 2), (4, 2), (5, 2), (6, 2),
        (3, 3), (4, 3), (5, 3), (6, 3),
        (4, 4), (5, 4), (6, 4),
        (5, 5), (6, 5),
        (6, 6),
    }

    def __init__(self, domino: t.Tuple[int, int], *args, **kwargs) -> None:
        super().__init__()
        head, tail = domino
        assert 0 <= head <= 6, f'Domino head is not between 0 and 6: {head}'
        assert 0 <= tail <= 6, f'Domino tail is not between 0 and 6: {tail}'
        assert head >= tail, f'Domino tail cannot be bigger than the head ({head} >= {tail} should be True)!'
        assert (head, tail) in GameDomino.dominoes, f'Domino not found in the valid dominoes: {(head, tail)}'
        self.__head: int = head
        self.__tail: int = tail

    def is_double_domino(self) -> bool:
        return self.head == self.tail

    @property
    def head(self) -> int:
        return self.__head

    @property
    def tail(self) -> int:
        return self.__tail

    def __hash__(self) -> int:
        return hash((self.head, self.tail))

    def __eq__(self, other: object) -> bool:
        if id(self) == id(other):
            return True
        if not isinstance(other, GameDomino):
            return False
        return self.head == other.head and self.tail == other.tail

    def __str__(self) -> str:
        return f'[{self.head}-{self.tail}]'


class GamePosition(object):
    def __init__(self,
                 *args,
                 index: t.Optional[t.Tuple[int, int]] = None,
                 pos: t.Optional[t.Tuple[str, str]] = None,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert index is not None or pos is not None, 'Both index and pos cannot be None.'
        assert index is None or pos is None, 'Both index and pos cannot be given.'

        if index is not None:
            self.index: t.Tuple[int, int] = index
        if pos is not None:
            self.index: t.Tuple[int, int] = GameBoard.pos_to_index(pos)

        assert 0 <= self.index[0] <= 14, f'Row index out of bounds: {self.index}'
        assert 0 <= self.index[1] <= 14, f'Column index out of bounds: {self.index}'

    def __sub__(self, other: 'GamePosition') -> t.Tuple[int, int]:
        return (self[0] - other[0], self[1] - other[1])

    def __setitem__(self, index: int, value: int) -> None:
        if index == 0:
            self.index = (value, self.index[1])
        elif index == 1:
            self.index = (self.index[0], value)
        else:
            raise ValueError('Index is not zero or one.')

    def __getitem__(self, index: int) -> int:
        return self.index[index]

    def __lt__(self, other: 'GamePosition') -> bool:
        if self[0] < other[0]:
            return True
        if self[0] > other[0]:
            return False
        if self[1] < other[1]:
            return True
        return False

    def __le__(self, other: 'GamePosition') -> bool:
        return not (other < self)

    def __eq__(self, other: object) -> bool:
        if id(self) == id(other):
            return True
        if not isinstance(other, GamePosition):
            return False
        return self[0] == other[0] and self[1] == other[1]

    def __hash__(self) -> int:
        return hash(self.index)

    def __str__(self) -> str:
        return f'{self.index[0]}:{self.index[1]}'

    @staticmethod
    def horizontal(pos1: 'GamePosition', pos2: 'GamePosition') -> bool:
        return pos1[0] == pos2[0]

    @staticmethod
    def vertical(pos1: 'GamePosition', pos2: 'GamePosition') -> bool:
        return pos1[1] == pos2[1]


class GameMove(object):
    def __init__(self,
                 main_score: int,
                 main_player: str,
                 other_score: int,
                 other_player: str,
                 domino: GameDomino,
                 positions: t.Tuple[GamePosition, GamePosition],
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Ensure that the input is valid
        assert np.abs(positions[0] - positions[1]).sum(axis=0) == 1, 'GameDominoMove cannot have positions on diagonals.'
        assert positions[0] != positions[1], 'GameDominoMove cannot have both positions to be the same.'
        assert main_player != other_player, 'Same player in the same move is not allowed twice.'
        assert main_player in ['player1', 'player2'], 'Invalid player number: {}'.format(main_player)
        assert other_player in ['player1', 'player2'], 'Invalid player number: {}'.format(other_player)
        assert main_score >= 0, 'Score cannot be negative: {}'.format(main_score)
        assert other_score >= 0, 'Score cannot be negative: {}'.format(other_score)

        # Initialize the internal attributes using the given params
        self.position: t.Tuple[GamePosition, GamePosition] = positions
        self.domino: GameDomino = domino
        self.main_player: str = main_player
        self.main_score: int = main_score
        self.other_player: str = other_player
        self.other_score: int = other_score

    def __str__(self) -> str:
        # Give an alias over the self instance
        pos: Tuple[Tuple[str, str], Tuple[str, str]] = GameBoard.index_to_pos(self.position[0].index), GameBoard.index_to_pos(self.position[1].index)
        head_pos: str = pos[0][0] + pos[0][1]
        tail_pos: str = pos[1][0] + pos[1][1]
        head_domino: int = self.domino.head
        tail_domino: int = self.domino.tail
        score: int = self.main_score

        # Ensure left-right, top-down order
        if self.position[0] <= self.position[1]:
            return f'{head_pos} {head_domino}{os.linesep}{tail_pos} {tail_domino}{os.linesep}{score}'
        else:
            return f'{tail_pos} {tail_domino}{os.linesep}{head_pos} {head_domino}{os.linesep}{score}'


class GameBoard(object):
    MAX_ROWS: int = 15
    MAX_COLS: int = 15
    diamond_to_points: t.Dict[str, int] = {
        'blue':   1,
        'yellow': 2,
        'brown':  3,
        'black':  4,
        'green':  5,
    }
    points_to_diamond: t.Dict[int, str] = {
        v: k for k, v in diamond_to_points.items()
    }

    def __init__(self,
                 grid: t.List[t.List['GameBoardCell']],
                 dominoes_on: t.Dict[GameDomino, t.List[t.Tuple[GamePosition, GamePosition]]],
                 dominoes_off: t.List[GameDomino], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dominoes_on: t.Dict[GameDomino, t.List[t.Tuple[GamePosition, GamePosition]]] = dominoes_on
        self.dominoes_off: t.List[GameDomino] = dominoes_off
        self.grid: t.List[t.List[GameBoardCell]] = grid

    def __add__(self, move: GameMove) -> 'GameBoard':
        # Minimize possible invalid moves by ensuring that a piece is always available when applying the respective move
        assert move.domino in self.dominoes_off, f'Domino is not available anymore: ({move.domino.head, move.domino.tail})!'
        assert len(self.dominoes_on.get(move.domino, [])) <= 2, f'Domino cannot be placed anymore: ({move.domino.head, move.domino.tail})!'

        # Copy the old grid in order to build a new one (immutable approach)
        new_game_board: GameBoard = copy.deepcopy(self)

        # Add the head of the domino to the first pos
        assert (cell := new_game_board.grid[move.position[0][0]][move.position[0][1]]).is_empty(), f'Tried to place domino on empty cell: {cell}'
        new_game_board[move.position[0][0]][move.position[0][1]] = GameBoardCell(
            diamond=new_game_board[move.position[0][0]][move.position[0][1]].diamond(),
            index=move.position[0].index,
            domino=move.domino.head,
        )

        # Add the tail of the domino to the second pos
        assert (cell := new_game_board.grid[move.position[1][0]][move.position[1][1]]).is_empty(), f'Tried to place domino on empty cell: {cell}'
        new_game_board[move.position[1][0]][move.position[1][1]] = GameBoardCell(
            diamond=new_game_board[move.position[1][0]][move.position[1][1]].diamond(),
            index=move.position[1].index,
            domino=move.domino.tail,
        )

        # Mark that the domino was placed on the board
        new_game_board.dominoes_off.remove(move.domino)
        new_game_board.dominoes_on[move.domino] = new_game_board.dominoes_on.get(move.domino, []) + [move.position]
        assert len(new_game_board.dominoes_on[move.domino]) <= 2, 'Too many dominoes of the same type were placed!'

        # Create a new grid with the new cells
        return new_game_board

    def __getitem__(self, index: int) -> t.List['GameBoardCell']:
        return self.grid[index]

    def __str__(self) -> str:
        output: str = ''
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                output += str(self.grid[i][j].value())
                if j != len(self.grid[i]) - 1:
                    output += ','
            if i != len(self.grid) - 1:
                output += '\n'
        return output

    def value(self) -> np.ndarray:
        return np.array([[self.grid[i][j].value() for j in range(len(self.grid[i]))] for i in range(len(self.grid))], dtype=np.int32)

    @staticmethod
    def neighbors(pos: t.Tuple[GamePosition, GamePosition]) -> t.Tuple[t.List[GamePosition | None], t.List[GamePosition | None]]:
        neighbors: t.Tuple[t.List[GamePosition | None], t.List[GamePosition | None]] = ([], [])

        # Clockwise manner for each head
        if GamePosition.horizontal(pos[0], pos[1]):
            indices: t.List[t.Tuple[int, int]] = [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0)]
        elif GamePosition.vertical(pos[0], pos[1]):
            indices: t.List[t.Tuple[int, int]] = [(0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)]
        else:
            raise Exception('Invalid domino positions, cannot find neighbors: {} {}-{}'.format(pos[0], pos[1]))

        # Compute the neighbour indices regardless if it's vertical or horizontal
        pos = pos if (keep_order := pos[0] < pos[1]) else (pos[1], pos[0])
        for i, j in indices:
            for index, sign in [(0, 1), (1, -1)]:
                row: int = pos[index][0] + i * sign
                col: int = pos[index][1] + j * sign
                if 0 <= row < GameBoard.MAX_ROWS and 0 <= col < GameBoard.MAX_COLS:
                    neighbors[index].append(GamePosition(index=(row, col)))
                else:
                    neighbors[index].append(None)
        return neighbors if keep_order else (neighbors[1], neighbors[0])

    @staticmethod
    def empty() -> 'GameBoard':
        # Create a matrix showing the positions of the diamonds
        diamonds: t.List[t.List[t.Optional[int]]] = [
            [   5, None, None,    4, None, None, None,    3, None, None, None,    4, None, None,    5],
            [None, None,    3, None, None,    4, None, None, None,    4, None, None,    3, None, None],
            [None,    3, None, None,    2, None, None, None, None, None,    2, None, None,    3, None],
            [   4, None, None,    3, None,    2, None, None, None,    2, None,    3, None, None,    4],
            [None, None,    2, None,    1, None,    1, None,    1, None,    1, None,    2, None, None],
            [None,    4, None,    2, None,    1, None, None, None,    1, None,    2, None,    4, None],
            [None, None, None, None,    1, None, None, None, None, None,    1, None, None, None, None],
            [   3, None, None, None, None, None, None, None, None, None, None, None, None, None,    3],
            [None, None, None, None,    1, None, None, None, None, None,    1, None, None, None, None],
            [None,    4, None,    2, None,    1, None, None, None,    1, None,    2, None,    4, None],
            [None, None,    2, None,    1, None,    1, None,    1, None,    1, None,    2, None, None],
            [   4, None, None,    3, None,    2, None, None, None,    2, None,    3, None, None,    4],
            [None,    3, None, None,    2, None, None, None, None, None,    2, None, None,    3, None],
            [None, None,    3, None, None,    4, None, None, None,    4, None, None,    3, None, None],
            [   5, None, None,    4, None, None, None,    3, None, None, None,    4, None, None,    5],
        ]

        # Create an empty game board using the diamonds
        grid: t.List[t.List[GameBoardCell]] = []
        for i in range(len(diamonds)):
            row: t.List[GameBoardCell] = []
            for j in range(len(diamonds[0])):
                row.append(GameBoardCell(index=(i, j), diamond=diamonds[i][j]))
            grid.append(row)

        # Create the list of possible dominoes
        dominoes_off: t.List[GameDomino] = [GameDomino(domino=domino) for domino in GameDomino.dominoes] * 2

        # Fill the board with the empty cells
        return GameBoard(grid=grid, dominoes_on=dict(), dominoes_off=dominoes_off)

    @staticmethod
    def index_to_row(index: int) -> str:
        assert 0 <= index <= 14, f'Row index is outside valid range: {index}'
        return str(index + 1)

    @staticmethod
    def row_to_index(row: str) -> int:
        assert 0 <= (index := int(row) - 1) <= 14, f'Row is outside valid range: {row}'
        return index

    @staticmethod
    def index_to_col(index: int) -> str:
        assert 0 <= index <= 14, f'Column index is outside valid range: {index}'
        return chr(ord('A') + index)

    @staticmethod
    def col_to_index(col: str) -> int:
        assert 0 <= (index := ord(col) - ord('A')) <= 14, f'Column is outside valid range: {col}'
        return index

    @staticmethod
    def index_to_pos(index: t.Tuple[int, int]) -> t.Tuple[str, str]:
        return GameBoard.index_to_row(index[0]), GameBoard.index_to_col(index[1])

    @staticmethod
    def pos_to_index(pos: t.Tuple[str, str]) -> t.Tuple[int, int]:
        return GameBoard.row_to_index(pos[0]), GameBoard.col_to_index(pos[1])


class GameBoardCell(object):
    def __init__(self,
                 index: t.Tuple[int, int],
                 *args,
                 domino: t.Optional[int] = None,
                 diamond: t.Optional[t.Union[str, int]] = None,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if domino is not None:
            assert 0 <= domino <= 6, f'Domino value is outside range: {domino}'

        if isinstance(diamond, int):
            assert diamond in GameBoard.points_to_diamond.keys(), f'Invalid diamond points: {diamond}'
            self.__diamond: t.Optional[str] = GameBoard.points_to_diamond[diamond]
        if isinstance(diamond, str):
            assert diamond in GameBoard.diamond_to_points.keys(), f'Invalid diamond type: {diamond}'
            self.__diamond: t.Optional[str] = diamond
        if diamond is None:
            self.__diamond = None

        self.__domino: t.Optional[int] = domino
        self.index: Tuple[int, int] = index

    def is_diamond(self) -> bool:
        return self.__diamond is not None

    def is_empty(self) -> bool:
        return self.__domino is None

    def diamond(self) -> t.Optional[int]:
        return None if self.__diamond is None else GameBoard.diamond_to_points[self.__diamond]

    def domino(self) -> t.Optional[int]:
        return None if self.__domino is None else self.__domino

    def value(self) -> int:
        if (domino := self.domino()) is None:
            return 7
        else:
            return domino

    def __str__(self) -> str:
        pos: Tuple[str, str] = GameBoard.index_to_pos(self.index)
        return f'({pos[0]}{pos[1]}, {self.diamond()}, {self.domino()})'

