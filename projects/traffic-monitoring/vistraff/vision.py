from .data import CarTrafficVideo
import pathlib as pb



class BackgroundSubtractor(object):
    def __init__(self, path: pb.Path | None = None, *args, **kwargs) -> None:
        super().__init__()
        self.path: pb.Path | None = path


