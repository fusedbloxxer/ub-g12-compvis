from sklearn.utils import compute_class_weight, compute_sample_weight
import torch
from torch.utils import data
from torch import Tensor
import pathlib as pb
import pandas as pd
import typing as t
import numpy as np
import cv2 as cv
import abc
import re

from .util import folder_exists, keep_aspect_ratio
from .vision import Board2GridOperation
from .preprocess import DataPreprocess


class Dataset(abc.ABC):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()


class GameDataset(Dataset):
    def __init__(self, path: pb.Path, task: str, train: bool = True) -> None:
        super().__init__()

        # Argument validation
        self.task: str = task
        self.train: bool = train
        assert self.task in ['bonus_task', 'regular_task'], f'Invalid task type, found {self.task}'

        # Ensure all given directories exist
        self.path: pb.Path = path
        self.path_input: pb.Path
        self.path_truth: t.Optional[pb.Path] = None
        for suffix in ['_input'] + ([] if not self.train else ['_truth']):
            assert folder_exists(self.path / suffix[1:] / self.task)
            setattr(self, 'path' + suffix, self.path / suffix[1:] / self.task)

        # Eagerly create the output dirs
        self.path_output: pb.Path = self.path / 'output' / self.task
        self.path_output.mkdir(parents=True, exist_ok=True)


class DDDHelpDataset(Dataset):
    def __init__(self, path: pb.Path, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.path_help: pb.Path = path
        assert folder_exists(self.path_help)
        self.pattern_image: str = '?*.jpg'
        self.regex_image: str = '(?P<image>[0-9]+).jpg'
        self.image_list: np.ndarray = np.arange(1, 16)

    def template(self,
                 index: int | None = None,
                 dynamic_retrieval: bool=False,
                 scale: float=1.0,
                 selection_type: str='pts',
                 template_type: str='board') -> np.ndarray:
        # By default the second helper image is the one used
        if index is None:
            index = 1

        # Retrieve an image by index, downscale and extract its template
        template_image: np.ndarray = self[index][0]
        template_image = cv.resize(template_image, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)

        # Make sure there is always something to fallback onto
        hardcoded_fallback_template: np.ndarray = self[1][0]
        hardcoded_fallback_template = cv.resize(hardcoded_fallback_template, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)

        # Handle two types of templates: board & grid
        if template_type == 'grid':
            relative_coordinates: np.ndarray = np.array([
                [0.27380952, 0.17953488],
                [0.73179272, 0.18139535],
                [0.73739496, 0.8       ],
                [0.27170868, 0.80651163]
            ])
        elif template_type == 'board':
            relative_coordinates: np.ndarray = np.array([
                [0.16946779, 0.04930233],
                [0.83263305, 0.05209302],
                [0.83963585, 0.93209302],
                [0.16386555, 0.94604651]
            ])
        else:
            raise ValueError(f'Invalid template type: {template_type}')
        hardcoded_fallback_pts: np.ndarray = relative_coordinates * np.flip(hardcoded_fallback_template.shape)
        fallback: bool = False

        # Extract the absolute x, y coordinates of the corners
        if index == 1 and not dynamic_retrieval:
            # Hardcoded relative values to the image shape
            template_corner_pts: np.ndarray = hardcoded_fallback_pts
        elif selection_type == 'roi':
            # Dynamic values selected by the user using the given template
            roi: np.ndarray = cv.selectROI('Template ROI', template_image, showCrosshair=True)
            cv.destroyWindow('Template ROI')
            if np.sum(roi) == 0:
                template_corner_pts: np.ndarray = hardcoded_fallback_pts
                fallback = True
            else:
                template_corner_pts: np.ndarray = np.array([
                    [roi[0],                   roi[1]],
                    [roi[0] + roi[2],          roi[1]],
                    [roi[0] + roi[2], roi[1] + roi[3]],
                    [roi[0],          roi[1] + roi[3]]
                ])
        else:
            # Select points by clicking on the image contents
            image: np.ndarray = template_image.copy()
            pts: t.List[t.List[int]] = []

            # Listen to click events
            def on_mouse_click(event, x, y, flags, params):
                if event == cv.EVENT_LBUTTONUP:
                    cv.circle(image, (x, y), 3, (0, 255, 255), 3)
                    pts.append([x, y])

            # Create the window only once
            cv.namedWindow('Template Points')

            # Select all corners
            while len(pts) < 4:
                cv.imshow('Template Points', image)
                cv.setMouseCallback('Template Points', on_mouse_click)

                # Exit on C key
                if cv.waitKey(1) & 0xFF == 99:
                    template_corner_pts: np.ndarray = hardcoded_fallback_pts
                    fallback = True
                    break

            # Clear to avoid leaks
            cv.destroyWindow('Template Points')
            template_corner_pts: np.ndarray = np.array(pts)

        # Check if the fallback has been triggered
        template_corner_pts = hardcoded_fallback_pts if fallback else template_corner_pts
        template_image = hardcoded_fallback_template if fallback else template_image

        # Compute relative x, y coordinates
        template_pts: np.ndarray = template_corner_pts / np.flip(template_image.shape)

        # Actual x, y coordinates
        template_pts *= np.flip(np.array(template_image.shape), axis=0)[None, ...]
        template_pts  = template_pts.astype(np.uint)

        # Length of each side
        tl_line = int(np.floor(np.sqrt(np.sum((template_pts[0] - template_pts[1]) ** 2))).astype(np.uint).item())
        tr_line = int(np.floor(np.sqrt(np.sum((template_pts[1] - template_pts[2]) ** 2))).astype(np.uint).item())
        br_line = int(np.floor(np.sqrt(np.sum((template_pts[2] - template_pts[3]) ** 2))).astype(np.uint).item())
        bl_line = int(np.floor(np.sqrt(np.sum((template_pts[3] - template_pts[0]) ** 2))).astype(np.uint).item())

        # Approximate maximum-side square
        mx_line = int(np.max([tl_line, tr_line, br_line, bl_line]))

        # Desired x, y coordinates
        desired_pts: np.ndarray = np.array([
            [      0,       0],
            [mx_line,       0],
            [mx_line, mx_line],
            [      0, mx_line]
        ])

        # Apply perspective transformation
        transform = cv.getPerspectiveTransform(template_pts.astype(np.float32), desired_pts.astype(np.float32))
        return cv.warpPerspective(template_image, transform, (mx_line, mx_line))

    def __getitem__(self, index: int | slice) -> np.ndarray:
        if isinstance(index, int):
            index_image: slice = slice(index, index + 1)
        else:
            index_image: slice = index

        image_list: t.List[np.ndarray] = []
        for image_path in sorted(self.path_help.glob(self.pattern_image), key=lambda x: x.name):
            match: re.Match[str] | None = re.search(self.regex_image, image_path.name)

            if not match:
                continue

            try:
                num: int
                if (num := int(match.group('image'))) not in self.image_list[index_image]:
                    continue
            except ValueError:
                continue

            image: np.ndarray = cv.imread(str(image_path.absolute()), cv.IMREAD_GRAYSCALE)
            image = keep_aspect_ratio(image, np.array([3_072, 4_080]))
            image_list.append(image)
        return np.stack(image_list, axis=0)

    def __len__(self) -> int:
        return len(list(self.path_help.glob(self.pattern_image)))


class DDDBonusGameDataset(GameDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, task='bonus_task', **kwargs)

        self.pattern_image: str = '?*.jpg'
        self.pattern_truth: str = '?*.txt'
        self.regex_image: str = '(?P<task>[0-9]+).jpg'
        self.regex_truth: str = '(?P<task>[0-9]+).txt'

        self.task_list: np.ndarray = np.arange(1, 11)

    def __getitem__(self, index: int | slice) -> t.Tuple[np.ndarray, pd.DataFrame]:
        if isinstance(index, int):
            task_index: slice = slice(index, index + 1)
        else:
            task_index: slice = index

        # Retrieve a single image
        image_list: t.List[np.ndarray] = []
        misplacements: pd.DataFrame = pd.DataFrame()
        for image_path in sorted(self.path_input.glob(self.pattern_image), key=lambda x: x.name):
            match: re.Match[str] | None = re.search(self.regex_image, image_path.name)

            if not match:
                continue

            try:
                task: int
                if (task := int(match.group('task'))) not in self.task_list[task_index]:
                    continue
            except ValueError:
                continue

            image: np.ndarray = cv.imread(str(image_path.absolute()), cv.IMREAD_GRAYSCALE)
            image = keep_aspect_ratio(image, np.array([3_072, 4_080]))
            image_list.append(image)

            contents: t.Dict[str, t.Any] = {
                'img_name': pd.Series(data=[f'{match.group("task")}.jpg'], dtype=pd.StringDtype()),
                'img_path': pd.Series(data=[self.path_input / f'{match.group("task")}.jpg'], dtype=pd.StringDtype())
            }

            df_misplacement = pd.DataFrame(contents)
            misplacements = pd.concat([misplacements, df_misplacement])
        misplacements.reset_index(inplace=True, drop=True)
        images: np.ndarray = np.stack(image_list, axis=0)

        # In the case of testing no labels are available
        if not self.train:
            return images, misplacements

        # Retrieve the labels with misplacements
        index = 0
        assert self.path_truth is not None
        for truth_path in sorted(self.path_truth.glob(self.pattern_truth), key=lambda x: x.name):
            match: re.Match[str] | None = re.search(self.regex_truth, truth_path.name)

            if not match:
                continue

            try:
                task: int
                if (task := int(match.group('task'))) not in self.task_list[task_index]:
                    continue
            except ValueError:
                continue

            with open(str(truth_path.absolute()), 'r') as label_file:
                count: int = int(label_file.readline().strip('\n \t'))
                entries: t.List[str] = [entry.strip('\n \t') for entry in label_file.readlines() if len(entry.strip('\n \t')) != 0]

                if len(entries) % 2 != 0:
                    raise ValueError('The label file format is wrong.')

                misplacement_list: t.List[t.Tuple[str, str]] = []
                for i in range(0, len(entries), 2):
                    misplacement_list.append((entries[i], entries[i + 1]))

                if 'count' not in misplacements:
                    misplacements.insert(0, column='count', value=pd.Series())
                    misplacements.fillna(0, inplace=True)
                if 'misplacements' not in misplacements:
                    misplacements.insert(1, column='misplacements', value=pd.Series(dtype='object'))

                # Insert the current values
                misplacements.at[index, 'count'] = int(count)
                misplacements.at[index, 'misplacements'] = misplacement_list
                index += 1
        return images, misplacements

    def __len__(self) -> int:
        return len(list(self.path_input.glob(self.pattern_image)))


class DDDRegularGameDataset(GameDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, task='regular_task', **kwargs)

        self.pattern_image: str = '?*_?*.jpg'
        self.pattern_truth: str = '?*_?*.txt'
        self.pattern_move:  str = '?*_moves.txt'
        self.regex_image: str = '(?P<game>[0-9]+)_(?P<move>[0-9]+).jpg'
        self.regex_truth: str = '(?P<game>[0-9]+)_(?P<move>[0-9]+).txt'
        self.regex_move:  str = '(?P<game>[0-9]+)_moves.txt'

        self.game_list: np.ndarray = np.array(sorted(list(set([int(x.name[:1]) for x in self.path_input.glob(self.pattern_image)]))))
        self.move_list: np.ndarray = np.arange(1, 21)

    def __getitem__(self, game_index: t.Tuple[int | slice, int | slice] | int | slice) -> t.Tuple[np.ndarray, pd.DataFrame]:
        if isinstance(game_index, int):
            game_index = (game_index, slice(None))
        if isinstance(game_index, slice):
            game_index = (game_index, slice(None))
        if isinstance(game_index[0], int):
            index_game: slice = slice(game_index[0], game_index[0] + 1)
        else:
            index_game: slice = game_index[0]
        if isinstance(game_index[1], int):
            index_move: slice = slice(game_index[1], game_index[1] + 1)
        else:
            index_move: slice = game_index[1]

        # Load the images in-memory
        image_list: t.List[np.ndarray] = []
        for image_path in sorted(self.path_input.glob(self.pattern_image), key=lambda x: x.name):
            match: re.Match[str] | None = re.search(self.regex_image, image_path.name)

            if not match:
                continue

            try:
                game: int; move: int
                if (game := int(match.group('game'))) not in self.game_list[index_game]:
                    continue
                if (move := int(match.group('move'))) not in self.move_list[index_move]:
                    continue
            except ValueError:
                continue

            image: np.ndarray = cv.imread(str(image_path.absolute()), cv.IMREAD_GRAYSCALE)
            image = keep_aspect_ratio(image, np.array([3_072, 4_080]))
            image_list.append(image)
        images: np.ndarray = np.stack(image_list, axis=0)

        # Load the moves in-memory
        moves: pd.DataFrame = pd.DataFrame()
        for move_path in sorted(self.path_input.glob(self.pattern_move), key=lambda x: x.name):
            match: re.Match[str] | None = re.search(self.regex_move, move_path.name)

            if not match:
                continue

            try:
                game: int
                if (game := int(match.group('game'))) not in self.game_list[index_game]:
                    continue
            except ValueError:
                continue

            # Aggregate all necessary info in a DataFrame
            df_move: pd.DataFrame = pd.read_csv(move_path.absolute(), delimiter=' ', header=None, names=['img_name', 'player_turn'])
            df_move['img_name'] = df_move['img_name'].astype(pd.StringDtype())
            df_move.insert(0, column='img_path', value=df_move['img_name'].apply(lambda image_name: self.path_input / image_name).astype(pd.StringDtype()))
            df_move.insert(0, column='game', value=np.full_like(df_move.shape[0], game))
            df_move.insert(1, column='move', value=self.move_list)
            df_move.insert(len(df_move.columns), column='player', value=df_move['player_turn'].map(lambda p: 1 if p == 'player1' else 2))
            df_move.drop(inplace=True, columns='player_turn')
            df_move = df_move.iloc[index_move]
            moves = pd.concat([moves, df_move])
        moves.reset_index(inplace=True, drop=True)

        # In the case of testing no labels are available
        if not self.train:
            return images, moves

        # Load the labels into the previous DataFrame for ease of use
        pts: t.List[int] = []
        num: t.List[t.Tuple[int, int]] = []
        pos: t.List[t.Tuple[str, str]] = []
        assert self.path_truth is not None
        for label_path in sorted(self.path_truth.glob(self.pattern_truth), key=lambda x: x.name):
            match: re.Match[str] | None = re.search(self.regex_truth, label_path.name)

            if not match:
                continue

            try:
                game: int; move: int
                if (game := int(match.group('game'))) not in self.game_list[index_game]:
                    continue
                if (move := int(match.group('move'))) not in self.move_list[index_move]:
                    continue
            except ValueError:
                continue

            with open(str(label_path.absolute()), 'r') as label_file:
                position_1, digit_1 = label_file.readline().strip('\n \t').split(sep=' ')
                position_2, digit_2 = label_file.readline().strip('\n \t').split(sep=' ')
                digits: t.Tuple[int, int] = (int(digit_1), int(digit_2))
                position: t.Tuple[str, str] = (position_1, position_2)
                points = int(label_file.readline())

            pts.append(points)
            num.append(digits)
            pos.append(position)
        moves.insert(len(moves.columns), column='pos', value=pos)
        moves.insert(len(moves.columns), column='num', value=num)
        moves.insert(len(moves.columns), column='pts', value=pts)
        return images, moves

    def __len__(self) -> int:
        return len(list(self.path_input.glob(self.pattern_move)))


class DDDGameDataset(Dataset):
    def __init__(self, path: pb.Path, *args, train: bool=True, **kwargs) -> None:
        super().__init__()

        # Internal parameters
        self.train: bool = train

        # Create helper datasets
        self.dataset_help    = DDDHelpDataset(path=path / '..' / 'help')
        self.dataset_bonus   = DDDBonusGameDataset(path=path, train=self.train)
        self.dataset_regular = DDDRegularGameDataset(path=path, train=self.train)


class DDDCellDataset(Dataset, data.Dataset):
    def __init__(self, path: pb.Path, *args, **kwargs) -> None:
        super().__init__()

        self.path: pb.Path = path
        self.path_labels: pb.Path = self.path / 'labels.csv'
        self.labels: pd.DataFrame = pd.read_csv(self.path_labels, index_col=False)
        self.labels = self.labels.drop('Unnamed: 0', axis=1)
        self.labels = self.labels.dropna().reset_index(drop=True)
        self.labels['label'] = self.labels['label'].astype(np.int64)

        if not self.path_labels.exists():
            raise FileNotFoundError(str(self.path_labels.absolute()))
        if not self.path_labels.is_file():
            raise Exception('Is not a file: {}'.format(str(self.path_labels.absolute())))

    def __getitem__(self, index: int | slice) -> t.Tuple[np.ndarray, np.ndarray]:
        if isinstance(index, int):
            index_data: t.List[int] | slice = [index]
        else:
            index_data: t.List[int] | slice = index

        # Select the relevant entries
        labels: np.ndarray = self.labels.iloc[index_data]['label'].to_numpy(dtype=np.int32)

        # Read all images in-memory
        image_paths: pd.Series = self.labels.iloc[index_data]['cell_image_path']
        image_list: t.List[np.ndarray] = []
        for image_path in image_paths:
            image: np.ndarray = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
            image = cv.resize(image, dsize=(45, 45), interpolation=cv.INTER_LINEAR)
            image_list.append(image)
        return np.stack(image_list, axis=0), labels

    def __len__(self) -> int:
        return len(self.labels)

    @staticmethod
    def create_cell_dataset(path: pb.Path, subset: DDDGameDataset, board2grid: Board2GridOperation, overwrite_imgs: bool=False, overwrite_csv: bool=False) -> None:
        # Consider a subset path
        cell_dataset_path: pb.Path = path
        if cell_dataset_path.exists() and not overwrite_imgs:
            print('Cell dataset already exists: {}'.format(path))
            return

        # Read all data
        cell_dataset: DDDGameDataset = subset
        cell_dataset_path.mkdir(parents=True, exist_ok=False)
        regular_images, regular_labels = cell_dataset.dataset_regular[:, :]
        bonus_images, bonus_labels = cell_dataset.dataset_bonus[:]

        # Conatenate all images & labels
        cell_images: np.ndarray   = np.concatenate([regular_images, bonus_images], axis=0)
        cell_labels: pd.DataFrame = pd.concat([regular_labels, bonus_labels], join='inner', ignore_index=True)
        cell_labels_df = pd.DataFrame(columns=['original_img_path', 'cell_image_path', 'label'])

        for i, (image, (_, labels)) in enumerate(zip(cell_images, cell_labels.iterrows())):
            # Break the image into a matrix of cells
            grid, grid_patch, grid_cells = board2grid(image)

            original_image_path: pb.Path = pb.Path(labels['img_path'])
            cell_image_dir_path: pb.Path = cell_dataset_path / f'grid_{i + 1}'
            cell_image_dir_path.mkdir(parents=True, exist_ok=True)

            cv.imwrite(str((cell_image_dir_path / 'grid.jpg').absolute()), grid)
            cv.imwrite(str((cell_image_dir_path / 'original.jpg').absolute()), image)
            cv.imwrite(str((cell_image_dir_path / 'grid_patch.jpg').absolute()), grid_patch)

            for j in range(len(grid_cells)):
                for k in range(len(grid_cells[0])):
                    digit_row: str = str(j + 1)
                    ascii_column: str = chr(ord('A') + k)
                    cell_image_path: pb.Path = cell_image_dir_path / f'cell_{digit_row}{ascii_column}.jpg'
                    cv.imwrite(str(cell_image_path.absolute()), grid_cells[j][k])
                    cell_labels_df.loc[len(cell_labels_df)] = [str(original_image_path), str(cell_image_path), None]

        # Create or overwrite labels file
        csv_path: pb.Path = cell_dataset_path / 'labels.csv'
        if not csv_path.exists() or overwrite_csv:
            cell_labels_df.to_csv(csv_path, sep=',', header=True)


class DDDCellDatasetTorch(Dataset, data.Dataset):
    def __init__(self, path: pb.Path, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cell_dataset = DDDCellDataset(path=path, *args, **kwargs)
        self.preprocess = DataPreprocess()

    def __getitem__(self, index: int | slice) -> t.Tuple[Tensor, Tensor]:
        # Load the image in-memory and apply simple preprocessing
        images, labels = self.cell_dataset[index]
        images = self.preprocess.forward(images)

        # When called with a single index it should avoid returning a batch dimension
        if isinstance(index, int):
            images = images.squeeze(0)
            labels = labels.item()

        # Always return tensors
        return images, torch.tensor(labels)

    def __len__(self) -> int:
        return len(self.cell_dataset)


def compute_sample_class_weights(labels: Tensor) -> t.Tuple[Tensor, Tensor]:
    classes: Tensor = labels.unique()
    class_weights = torch.tensor(compute_class_weight('balanced', classes=classes.numpy(), y=labels.numpy()), dtype=torch.float32)
    sample_weights = torch.tensor(compute_sample_weight('balanced', y=labels.numpy()), dtype=torch.float32)
    return sample_weights, class_weights

