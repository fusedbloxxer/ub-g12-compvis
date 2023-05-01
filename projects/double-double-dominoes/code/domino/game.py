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
                 classifier_path: pb.Path,
                 *args,
                 template_index: t.Optional[int] = None,
                 template_selection_type: str = 'pts',
                 template_dynamic_retrieval: bool=False,
                 show_matrix: bool = False,
                 show_image: bool = False,
                 cell_splitter: str='hough',
                 cell_classifier: str = 'resnet',
                 train: bool = False,
                 **kwargs) -> None:
        super().__init__(*args, game_dataset=DDDGameDataset(dataset_path, train=train), **kwargs)

        # Use a pretrained model to classify the cell contents
        if cell_classifier == 'resnet':
            self.cell_classifier: CellClassifier = PretrainedCellClassifier.load_checkpoint(classifier_path, num_workers=0)
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

    def solve(self) -> None:
        # Solve the two tasks and optionally save the results in the their respective format
        self.task_regular.solve()
        self.task_bonus.solve()


class RegularTask(Task[DDDRegularGameDataset]):
    def __init__(self, dataset: DDDRegularGameDataset, board2matrix: Board2MatrixOpeation, *args, **kwargs) -> None:
        super().__init__(*args, game_dataset=dataset, **kwargs)
        self.board2matrix: Board2MatrixOpeation = board2matrix

    def solve(self) -> None:
        for game_index in range(len(self.dataset)):
            # Read one game at a time
            game_images, game_moves = self.dataset[game_index]

            # Build a single instance of the game to isolate states
            game = Game(index=game_index + 1,
                        images=game_images,
                        moves=game_moves,
                        board2matrix=self.board2matrix)

            # Solve the game
            game.solve(self.dataset.path_output)


class BonusTask(Task[DDDBonusGameDataset]):
    def __init__(self, dataset: DDDBonusGameDataset, board2matrix: Board2MatrixOpeation, *args, **kwargs) -> None:
        super().__init__(*args, game_dataset=dataset, **kwargs)
        self.board2matrix: Board2MatrixOpeation = board2matrix

    def solve(self) -> None:
        pass


class Game(object):
    def __init__(self,
                 index: int,
                 images: np.ndarray,
                 moves: pd.DataFrame,
                 board2matrix: Board2MatrixOpeation, *args, **kwargs) -> None:
        super().__init__()

        # Game specific content
        self.b2m: Board2MatrixOpeation = board2matrix
        self.player_moves: pd.DataFrame = moves
        self.game_states: t.List[GameState] = []
        self.game_moves: t.List[GameMove] = []
        self.images: np.ndarray = images
        self.index: int = index

    def solve(self, output: pb.Path) -> None:
        # Initialize the GameState from scratch as it's the first turn
        self.game_states: t.List[GameState] = [GameState.empty()]

        # Iterate through each image and the respective player move
        for image, (_, move) in zip(self.images, self.player_moves.iterrows()):
            # Image -> Matrix with possible bad predictions
            matrix: np.ndarray = self.b2m(image)

            # Combine the new changes with the previous state
            next_game_state: GameState = self.game_states[-1] + (matrix, move)

            # Go to the next state and retrieve the next move
            self.game_states.append(next_game_state)

        # Extract the list of moves that have occurred
        self.game_moves = [state.game_move for state in self.game_states if state.game_move is not None]

        # Save the results according to the established format as .txt files
        for i, move in enumerate(self.game_moves):
            with open(str(output / f'{self.index}_{i + 1:02}.txt'), 'w') as move_file:
                move_file.write(str(move))
                move_file.flush()


class GameState(object):
    def __init__(self, game_board: 'GameBoard', score_track: 'GameScoreTrack', *args, game_move: t.Optional['GameMove'] = None, **kwargs) -> None:
        super().__init__()
        self.game_board: GameBoard = game_board
        self.score_track: GameScoreTrack = score_track
        self.game_move: t.Optional[GameMove] = game_move

    def __add__(self, change: t.Tuple[np.ndarray, pd.Series]) -> 'GameState':
        # Unpack the change to extract the predicted cell values and leverage the player turn
        matrix, player_move = change

        # Start robust filtering of invalid predictions
        domino, pos = self.matrix2move(matrix)

        # Compute the score using the rules of the game
        scores: Tuple[int, int] = self.move2scores(domino, pos)

        # Find out who the main and other player are
        main_player: str  = 'player' + str(player_move.loc['player'])
        other_player: str = 'player' + str(GameState.other_player(main_player))

        # Create the game move using the computed score and placement
        game_move: GameMove = GameMove(main_score=scores[0], main_player=main_player, other_score=scores[1], other_player=other_player, domino=domino, positions=pos)
        new_score_track: GameScoreTrack = self.score_track + game_move
        new_board: GameBoard = self.game_board + game_move

        # Create a new game state using immutability
        return GameState(game_board=new_board, score_track=new_score_track, game_move=game_move)

    def move2scores(self, domino: 'GameDomino', pos: t.Tuple['GamePosition', 'GamePosition']) -> t.Tuple[int, int]:
        return 0, 0

    def matrix2move(self, matrix: np.ndarray) -> t.Tuple['GameDomino', t.Tuple['GamePosition', 'GamePosition']]:
        return (None, None)

    @staticmethod
    def other_player(player: str) -> str:
        if player == 'player1':
            return 'player2'
        elif player == 'player2':
            return 'player1'
        else:
            raise ValueError('Invalid player number: {}'.format(player))

    @staticmethod
    def empty() -> 'GameState':
        # Initialize the first game state (beginning of the game)
        game_board: GameBoard = GameBoard.empty()
        score_track: GameScoreTrack = GameScoreTrack.empty()
        return GameState(game_board=game_board, score_track=score_track, game_move=None)


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
                 dominoes_on: t.Dict[GameDomino, t.Tuple[GamePosition, GamePosition]],
                 dominoes_off: t.Set[GameDomino], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dominoes_on: t.Dict[GameDomino, t.Tuple[GamePosition, GamePosition]] = dominoes_on
        self.dominoes_off: t.Set[GameDomino] = dominoes_off
        self.grid: t.List[t.List[GameBoardCell]] = grid

    def __add__(self, move: GameMove) -> 'GameBoard':
        # Minimize possible invalid moves by ensuring that a piece is always available when applying the respective move
        assert move.domino in self.dominoes_off, f'Domino is not available anymore: ({move.domino.head, move.domino.tail})!'
        assert move.domino not in self.dominoes_on, f'Domino is already placed: ({move.domino.head, move.domino.tail})!'

        # Copy the old grid in order to build a new one (immutable approach)
        new_game_board: GameBoard = copy.deepcopy(self)

        # Add the head of the domino to the first pos
        assert (cell := new_game_board.grid[move.position[0].index[0]][move.position[0].index[1]]).is_empty(), f'Tried to place domino on empty cell: {cell}'
        new_game_board.grid[move.position[0].index[0]][move.position[0].index[1]] = GameBoardCell(
            index=move.position[0].index,
            domino=move.domino.head,
        )

        # Add the tail of the domino to the second pos
        assert (cell := new_game_board.grid[move.position[1].index[0]][move.position[1].index[1]]).is_empty(), f'Tried to place domino on empty cell: {cell}'
        new_game_board.grid[move.position[1].index[0]][move.position[1].index[1]] = GameBoardCell(
            index=move.position[1].index,
            domino=move.domino.tail,
        )

        # Mark that the domino was placed on the board
        new_game_board.dominoes_off.discard(move.domino)
        new_game_board.dominoes_on[move.domino] = move.position

        # Create a new grid with the new cells
        return new_game_board

    def value(self) -> np.ndarray:
        return np.array([[self.grid[i][j].value() for j in range(len(self.grid[i]))] for i in range(len(self.grid))], dtype=np.int32)

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
        dominoes_off: t.Set[GameDomino] = {GameDomino(domino=domino) for domino in GameDomino.dominoes}

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
        return self.domino() or 7

    def __str__(self) -> str:
        pos: Tuple[str, str] = GameBoard.index_to_pos(self.index)
        return f'({pos[0]}{pos[1]}, {self.diamond()}, {self.domino()})'

