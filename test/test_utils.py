import importlib
import os
import pathlib

import numpy as np
import pytest

import torch


def in_fbcode() -> bool:
    return os.environ.get("IN_FBCODE_TORCHCODEC") == "1"


def assert_equal(*args, **kwargs):
    torch.testing.assert_close(*args, **kwargs, atol=0, rtol=0)


def get_video_path(filename: str) -> pathlib.Path:
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


def get_reference_video_path() -> pathlib.Path:
    return get_video_path("nasa_13013.mp4")


def get_reference_audio_path() -> pathlib.Path:
    return get_video_path("nasa_13013.mp4.audio.mp3")


def load_tensor_from_file(filename: str) -> torch.Tensor:
    file_path = get_video_path(filename)
    return torch.load(file_path, weights_only=True)


def get_reference_video_tensor() -> torch.Tensor:
    arr = np.fromfile(get_reference_video_path(), dtype=np.uint8)
    video_tensor = torch.from_numpy(arr)
    return video_tensor


@pytest.fixture()
def reference_video_tensor() -> torch.Tensor:
    return get_reference_video_tensor()
