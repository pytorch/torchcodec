import importlib
import os
import pathlib

from dataclasses import dataclass
from typing import Dict

import numpy as np

import torch


# For use with decoded data frames, or in other instances were we are confident that
# reference and test tensors should be exactly equal. This is true for decoded data
# frames from media because we expect our decoding to exactly match what a user can
# do on the command line with ffmpeg.
def assert_tensor_equal(*args, **kwargs):
    torch.testing.assert_close(*args, **kwargs, atol=0, rtol=0)


# For use with floating point metadata, or in other instances where we are not confident
# that reference and test tensors can be exactly equal. This is true for pts and duration
# in seconds, as the reference values are from ffprobe's JSON output. In that case, it is
# limiting the floating point precision when printing the value as a string. The value from
# JSON and the value we retrieve during decoding are not exactly the same.
def assert_tensor_close(*args, **kwargs):
    torch.testing.assert_close(*args, **kwargs, atol=1e-6, rtol=1e-6)


def in_fbcode() -> bool:
    return os.environ.get("IN_FBCODE_TORCHCODEC") == "1"


def _get_file_path(filename: str) -> pathlib.Path:
    if in_fbcode():
        resource = (
            importlib.resources.files(__spec__.parent)
            .joinpath("resources")
            .joinpath(filename)
        )
        with importlib.resources.as_file(resource) as path:
            return path
    else:
        return pathlib.Path(__file__).parent / "resources" / filename


def _load_tensor_from_file(filename: str) -> torch.Tensor:
    file_path = _get_file_path(filename)
    return torch.load(file_path, weights_only=True).permute(2, 0, 1)


@dataclass
class TestFrameInfo:
    pts_seconds: float
    duration_seconds: float


@dataclass
class TestContainerFile:
    filename: str
    frames: Dict[int, TestFrameInfo]

    @property
    def path(self) -> pathlib.Path:
        return _get_file_path(self.filename)

    def to_tensor(self) -> torch.Tensor:
        arr = np.fromfile(self.path, dtype=np.uint8)
        return torch.from_numpy(arr)

    def get_frame_data_by_index(self, idx: int) -> torch.Tensor:
        return _load_tensor_from_file(f"{self.filename}.frame{idx:06d}.pt")

    def get_frame_data_by_range(
        self, start: int, stop: int, step: int = 1
    ) -> torch.Tensor:
        tensors = [self.get_frame_data_by_index(i) for i in range(start, stop, step)]
        return torch.stack(tensors)

    def get_pts_seconds_by_range(
        self, start: int, stop: int, step: int = 1
    ) -> torch.Tensor:
        all_pts = [self.frames[i].pts_seconds for i in range(start, stop, step)]
        return torch.tensor(all_pts, dtype=torch.float64)

    def get_duration_seconds_by_range(
        self, start: int, stop: int, step: int = 1
    ) -> torch.Tensor:
        all_durations = [
            self.frames[i].duration_seconds for i in range(start, stop, step)
        ]
        return torch.tensor(all_durations, dtype=torch.float64)

    def get_frame_by_name(self, name: str) -> torch.Tensor:
        return _load_tensor_from_file(f"{self.filename}.{name}.pt")

    @property
    def empty_pts_seconds(self) -> torch.Tensor:
        return torch.empty([0], dtype=torch.float64)

    @property
    def empty_duration_seconds(self) -> torch.Tensor:
        return torch.empty([0], dtype=torch.float64)


@dataclass
class TestVideo(TestContainerFile):
    """
    Represents a video file used in our testing.

    Note that right now, we implicitly only support a single stream. Our current tests always
    use the "best" stream as defined by FFMPEG. In general, height, width and num_color_channels
    can vary per-stream. When we start testing multiple streams in the same video, we will have
    to generalize this class.
    """

    height: int
    width: int
    num_color_channels: int

    @property
    def empty_chw_tensor(self) -> torch.Tensor:
        return torch.empty(
            [0, self.num_color_channels, self.height, self.width], dtype=torch.uint8
        )


NASA_VIDEO = TestVideo(
    filename="nasa_13013.mp4",
    height=270,
    width=480,
    num_color_channels=3,
    # TODO_OPEN_ISSUE Scott: improve the testing framework so that these values are loaded from a JSON
    # file and not hardcoded. These values were copied over by hand from the JSON
    # output from the following command:
    #  $ ffprobe -v error -hide_banner -select_streams v:1 -show_frames -of json test/resources/nasa_13013.mp4 > out.json
    frames={
        0: TestFrameInfo(pts_seconds=0.0, duration_seconds=0.033367),
        1: TestFrameInfo(pts_seconds=0.033367, duration_seconds=0.033367),
        2: TestFrameInfo(pts_seconds=0.066733, duration_seconds=0.033367),
        3: TestFrameInfo(pts_seconds=0.100100, duration_seconds=0.033367),
        4: TestFrameInfo(pts_seconds=0.133467, duration_seconds=0.033367),
        5: TestFrameInfo(pts_seconds=0.166833, duration_seconds=0.033367),
        6: TestFrameInfo(pts_seconds=0.200200, duration_seconds=0.033367),
        7: TestFrameInfo(pts_seconds=0.233567, duration_seconds=0.033367),
        8: TestFrameInfo(pts_seconds=0.266933, duration_seconds=0.033367),
        9: TestFrameInfo(pts_seconds=0.300300, duration_seconds=0.033367),
        10: TestFrameInfo(pts_seconds=0.333667, duration_seconds=0.033367),
    },
)

# When we start actually decoding audio-only files, we'll probably need to define
# a TestAudio class with audio specific values. Until then, we only need a filename.
NASA_AUDIO = TestContainerFile(filename="nasa_13013.mp4.audio.mp3", frames={})
