import torch
import torch.utils.data as data
import pathlib as pb
import pandas as pd
import abc
from abc import ABC
import typing as t
import re
import cv2 as cv
import numpy as np


class CarTrafficVideoFrame(t.TypedDict):
    tensor: torch.Tensor
    image: np.ndarray
    size: np.ndarray
    total: int
    fps: int


class CarTrafficContextDict(CarTrafficVideoFrame):
    context: int


class CarTrafficTaskOneDict(t.TypedDict):
    tensor: torch.Tensor
    image: np.ndarray
    query: t.List[int]
    context: int
    example: int


class CarTrafficTaskTwoDict(CarTrafficContextDict):
    bbox: torch.Tensor


class CarTrafficTaskThreeDict(CarTrafficContextDict):
    pass


class CarTrafficVideo(object):
    def __init__(self, path: pb.Path, display: bool=False, *args, device: t.Optional[torch.device] = None, **kwargs) -> None:
        super().__init__()

        if device is None:
            device = torch.device('cpu')

        self.path: pb.Path = path
        self.display: bool = display
        self.device: torch.device = device

    def __iter__(self) -> t.Generator[CarTrafficVideoFrame, None, None]:
        video = cv.VideoCapture(str(self.path))

        if not video.isOpened():
            raise Exception('Video at path: {} could not be opened!'.format(self.path))

        while video.isOpened():
            has_frame, frame = video.read()

            if not has_frame:
                break

            frame_data: CarTrafficVideoFrame = {
                'tensor': torch.from_numpy(cv.cvtColor(frame.copy(), cv.COLOR_BGR2RGB)).to(self.device),
                'image': frame,
                'size': np.array([video.get(cv.CAP_PROP_FRAME_HEIGHT), video.get(cv.CAP_PROP_FRAME_WIDTH)], dtype=np.int32),
                'total': int(video.get(cv.CAP_PROP_FRAME_COUNT)),
                'fps': int(video.get(cv.CAP_PROP_FPS)),
            }

            if self.display:
                cv.imshow('Playing {} Video'.format(self.path.absolute()), frame)
                if cv.waitKey(int((1 / frame_data['fps']) * 1_000)) == ord('q'):
                    break

            yield frame_data

        # Free resources
        if self.display:
            cv.destroyWindow('Playing {} Video'.format(self.path.absolute()))
        video.release()


class CarTrafficBaseDataset(abc.ABC, data.Dataset):
    def __init__(self, path: pb.Path, *args, **kwargs) -> None:
        super().__init__()
        self.path: pb.Path = path

    @abc.abstractmethod
    def __getitem__(self, index: int) -> t.Any:
        raise NotImplementedError()

    @abc.abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()


class CarTrafficVideoDataset(CarTrafficBaseDataset):
    def __init__(self, path: pb.Path, *args, **kwargs) -> None:
        super().__init__(path, *args, **kwargs)

        # Expected format
        pattern_video: re.Pattern[str] = re.compile(r"(?P<context>\d\d).mp4")

        # Store the paths to each file for easy access
        self.lookup = pd.DataFrame(columns=['context', 'video_path'])

        # Read all files
        for video_path in sorted(path.iterdir(), key=lambda x: x.name):
            if not (match := pattern_video.fullmatch(video_path.name)):
                continue
            self.lookup.loc[len(self.lookup)] = [
                int(match.group('context')),
                str(video_path.relative_to('.'))
            ]

    def __getitem__(self, index: int) -> t.Generator[CarTrafficContextDict, None, None]:
        entry: pd.Series = self.lookup.iloc[index]
        return ({ **frame, 'context': entry['context'] } for frame in CarTrafficVideo(entry['video_path']))

    def __len__(self) -> int:
        return len(self.lookup)


class CarTrafficTaskOneDataset(CarTrafficBaseDataset):
    def __init__(self, path: pb.Path, *args, **kwargs) -> None:
        super().__init__(path, *args, **kwargs)

        # Expected formats
        pattern_common: re.Pattern[str] = re.compile(r"(?P<context>\d\d)_(?P<example>\d)")
        pattern_query: re.Pattern[str] = re.compile(rf"{pattern_common.pattern}_query.txt")
        pattern_image: re.Pattern[str] = re.compile(rf"{pattern_common.pattern}.jpg")

        # Store the paths to each file for easy access
        self.lookup = pd.DataFrame(columns=['context', 'example', 'image_path', 'query_path'])

        # Read all files
        image_paths: t.List[pb.Path] = []
        query_paths: t.List[pb.Path] = []
        for item_path in sorted(path.iterdir(), key=lambda x: x.name):
            if pattern_image.fullmatch(item_path.name):
                image_paths.append(item_path)
                continue
            if pattern_query.fullmatch(item_path.name):
                query_paths.append(item_path.relative_to('.'))
                continue
            continue

        for image_path, query_path in zip(image_paths, query_paths):
            image_name = pattern_common.search(image_path.name)
            query_name = pattern_common.search(query_path.name)

            self.lookup.loc[len(self.lookup)] = [
                int(image_name.group('context')),
                int(image_name.group('example')),
                str(image_path.relative_to('.')),
                str(query_path.relative_to('.')),
            ]

    def __getitem__(self, index: int) -> CarTrafficTaskOneDict:
        item: pd.Series[t.Any] = self.lookup.iloc[index]

        image_raw: np.ndarray = cv.imread(str(item['image_path']))
        image_proc: np.ndarray | torch.Tensor = image_raw.copy()
        image_proc = cv.cvtColor(image_proc, cv.COLOR_BGR2RGB)
        image_proc = torch.from_numpy(image_proc.transpose((2, 0, 1)))

        with open(item['query_path']) as query_file:
            queries: t.List[int] = [int(line.strip()) for line in query_file.readlines()[1:]]

        return {
            'query': queries,
            'image': image_raw,
            'tensor': image_proc,
            'context': item['context'],
            'example': item['example'],
        }

    def __len__(self) -> int:
        return len(self.lookup)


class CarTrafficTaskTwoDataset(CarTrafficBaseDataset):
    def __init__(self, path: pb.Path, *args, **kwargs) -> None:
        super().__init__(path, *args, **kwargs)

        # Expected format
        pattern_init_bbox: re.Pattern[str] = re.compile(r"(?P<context>\d\d).txt")

        # Store the paths to each file for easy access
        lookup: pd.DataFrame = pd.DataFrame(columns=['context', 'bbox'])

        # Consider the associated video data
        self.video_dataset = CarTrafficVideoDataset(path=path)

        # Read all files
        for bbox_path in sorted(path.iterdir(), key=lambda x: x.name):
            if not (match := pattern_init_bbox.fullmatch(bbox_path.name)):
                continue
            context = int(match.group('context'))
            lookup.loc[len(lookup)] = [
                context,
                str(bbox_path.relative_to('.'))
            ]

        # Join the two lookup using the context
        self.video_dataset.lookup = pd.merge(lookup, self.video_dataset.lookup, on=['context'], how='right')

    @property
    def lookup(self) -> pd.DataFrame:
        return self.video_dataset.lookup

    def __getitem__(self, index: int) -> t.Generator[CarTrafficTaskTwoDict, None, None]:
        entry: pd.Series = self.lookup.iloc[index]

        # Fetch the initial bbox
        if not pd.isnull(entry['bbox']):
            with open(entry['bbox']) as bbox_file:
                bbox = torch.tensor(list(map(lambda x: int(x), bbox_file.readlines()[1].strip().split(' ')[1:])), dtype=torch.int32)
        else:
            bbox = torch.zeros((4,), dtype=torch.int32)

        # Open the video stream
        return ({ **frame, 'bbox': bbox } for frame in self.video_dataset[index])

    def __len__(self) -> int:
        return len(self.lookup)


class CarTrafficTaskThreeDataset(CarTrafficBaseDataset):
    def __init__(self, path: pb.Path, *args, **kwargs) -> None:
        super().__init__(path, *args, **kwargs)

        # Consider the associated video data
        self.video_dataset = CarTrafficVideoDataset(path=path)

    @property
    def lookup(self) -> pd.DataFrame:
        return self.video_dataset.lookup

    def __getitem__(self, index: int) -> t.Generator[CarTrafficTaskThreeDict, None, None]:
        return self.video_dataset[index]

    def __len__(self) -> int:
        return len(self.video_dataset)


class CarTrafficDataset(object):
    def __init__(self, path: pb.Path, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Initialize all task datasets
        self.context = CarTrafficVideoDataset(path=path / 'context_videos_all_tasks', *args, **kwargs)
        self.task_3 = CarTrafficTaskThreeDataset(path=path / 'Task3', *args, **kwargs)
        self.task_2 = CarTrafficTaskTwoDataset(path=path / 'Task2', *args, **kwargs)
        self.task_1 = CarTrafficTaskOneDataset(path=path / 'Task1', *args, **kwargs)
