import importlib
import os
import pathlib

from dataclasses import dataclass

import numpy as np

import torch


def assert_tensor_equal(*args, **kwargs):
    torch.testing.assert_close(*args, **kwargs, atol=0, rtol=0)


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
    return torch.load(file_path, weights_only=True)


@dataclass
class TestContainerFile:
    filename: str

    @property
    def path(self) -> pathlib.Path:
        return _get_file_path(self.filename)

    def to_tensor(self) -> torch.Tensor:
        arr = np.fromfile(self.path, dtype=np.uint8)
        return torch.from_numpy(arr)

    def get_tensor_by_index(self, idx: int) -> torch.Tensor:
        return _load_tensor_from_file(f"{self.filename}.frame{idx:06d}.pt")

    def get_stacked_tensor_by_range(
        self, start: int, stop: int, step: int = 1
    ) -> torch.Tensor:
        tensors = [self.get_tensor_by_index(i) for i in range(start, stop, step)]
        return torch.stack(tensors)

    def get_tensor_by_name(self, name: str) -> torch.Tensor:
        return _load_tensor_from_file(f"{self.filename}.{name}.pt")


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
    def empty_hwc_tensor(self) -> torch.Tensor:
        return torch.empty(
            [0, self.height, self.width, self.num_color_channels], dtype=torch.uint8
        )


NASA_VIDEO = TestVideo(
    filename="nasa_13013.mp4", height=270, width=480, num_color_channels=3
)

# When we start actually decoding audio-only files, we'll probably need to define
# a TestAudio class with audio specific values. Until then, we only need a filename.
NASA_AUDIO = TestContainerFile(filename="nasa_13013.mp4.audio.mp3")
