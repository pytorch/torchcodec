import importlib
import os
import pathlib
import sys

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pytest

import torch

from torchcodec.decoders._core import get_ffmpeg_library_versions


# Decorator for skipping CUDA tests when CUDA isn't available. The tests are
# effectively marked to be skipped in pytest_collection_modifyitems() of
# conftest.py
def needs_cuda(test_item):
    return pytest.mark.needs_cuda(test_item)


def cpu_and_cuda():
    return ("cpu", pytest.param("cuda", marks=pytest.mark.needs_cuda))


def get_ffmpeg_major_version():
    return int(get_ffmpeg_library_versions()["ffmpeg_version"].split(".")[0])


# For use with decoded data frames. On CPU Linux, we expect exact, bit-for-bit
# equality. On CUDA Linux, we expect a small tolerance.
# On other platforms (e.g. MacOS), we also allow a small tolerance. FFmpeg does
# not guarantee bit-for-bit equality across systems and architectures, so we
# also cannot. We currently use Linux on x86_64 as our reference system.
def assert_frames_equal(*args, **kwargs):
    if sys.platform == "linux":
        if args[0].device.type == "cuda":
            atol = 2
            if get_ffmpeg_major_version() == 4:
                assert_tensor_close_on_at_least(
                    args[0], args[1], percentage=95, atol=atol
                )
            else:
                torch.testing.assert_close(*args, **kwargs, atol=atol, rtol=0)
        else:
            torch.testing.assert_close(*args, **kwargs, atol=0, rtol=0)
    else:
        torch.testing.assert_close(*args, **kwargs, atol=3, rtol=0)


# Asserts that at least `percentage`% of the values are within the absolute tolerance.
# Percentage is expected in [0, 100] (actually, [60, 100])
def assert_tensor_close_on_at_least(actual_tensor, ref_tensor, *, percentage, atol):
    # In theory lower bound should be 0, but we want to make sure we don't
    # mistakenly pass percentage in [0, 1]
    assert 60 < percentage <= 100, (
        f"Percentage must be in [60, 100], got {percentage}. "
        "Are you sure setting such a low tolerance is desired?"
    )
    assert (
        actual_tensor.device == ref_tensor.device
    ), f"Devices don't match: {actual_tensor.device} vs {ref_tensor.device}"

    abs_diff = (ref_tensor.float() - actual_tensor.float()).abs()
    valid_percentage = (abs_diff <= atol).float().mean() * 100
    if valid_percentage < percentage:
        raise AssertionError(
            f"Expected at least {percentage}% of values to be within atol={atol}, "
            f"but only {valid_percentage}% were."
        )


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

    # {stream_index -> {frame_index -> frame_info}}
    frames: Dict[int, Dict[int, TestFrameInfo]]

    default_stream_index: int

    @property
    def path(self) -> pathlib.Path:
        return _get_file_path(self.filename)

    def to_tensor(self) -> torch.Tensor:
        arr = np.fromfile(self.path, dtype=np.uint8)
        return torch.from_numpy(arr)

    def get_frame_data_by_index(
        self, idx: int, *, stream_index: Optional[int] = None
    ) -> torch.Tensor:
        if stream_index is None:
            stream_index = self.default_stream_index

        return _load_tensor_from_file(
            f"{self.filename}.stream{stream_index}.frame{idx:06d}.pt"
        )

    def get_frame_data_by_range(
        self,
        start: int,
        stop: int,
        step: int = 1,
        *,
        stream_index: Optional[int] = None,
    ) -> torch.Tensor:
        tensors = [
            self.get_frame_data_by_index(i, stream_index=stream_index)
            for i in range(start, stop, step)
        ]
        return torch.stack(tensors)

    def get_pts_seconds_by_range(
        self,
        start: int,
        stop: int,
        step: int = 1,
        *,
        stream_index: Optional[int] = None,
    ) -> torch.Tensor:
        if stream_index is None:
            stream_index = self.default_stream_index

        all_pts = [
            self.frames[stream_index][i].pts_seconds for i in range(start, stop, step)
        ]
        return torch.tensor(all_pts, dtype=torch.float64)

    def get_duration_seconds_by_range(
        self,
        start: int,
        stop: int,
        step: int = 1,
        *,
        stream_index: Optional[int] = None,
    ) -> torch.Tensor:
        if stream_index is None:
            stream_index = self.default_stream_index

        all_durations = [
            self.frames[stream_index][i].duration_seconds
            for i in range(start, stop, step)
        ]
        return torch.tensor(all_durations, dtype=torch.float64)

    def get_frame_info(
        self, idx: int, *, stream_index: Optional[int] = None
    ) -> TestFrameInfo:
        if stream_index is None:
            stream_index = self.default_stream_index

        return self.frames[stream_index][idx]

    @property
    def empty_pts_seconds(self) -> torch.Tensor:
        return torch.empty([0], dtype=torch.float64)

    @property
    def empty_duration_seconds(self) -> torch.Tensor:
        return torch.empty([0], dtype=torch.float64)


@dataclass
class TestVideoStreamInfo:
    width: int
    height: int
    num_color_channels: int


@dataclass
class TestVideo(TestContainerFile):
    stream_infos: Dict[int, TestVideoStreamInfo]

    @property
    def width(self) -> int:
        return self.stream_infos[self.default_stream_index].width

    @property
    def height(self) -> int:
        return self.stream_infos[self.default_stream_index].height

    @property
    def num_color_channels(self) -> int:
        return self.stream_infos[self.default_stream_index].num_color_channels

    @property
    def empty_chw_tensor(self) -> torch.Tensor:
        return torch.empty(
            [0, self.num_color_channels, self.height, self.width], dtype=torch.uint8
        )

    def get_width(self, *, stream_index: Optional[int]) -> int:
        if stream_index is None:
            stream_index = self.default_stream_index

        return self.stream_infos[stream_index].width

    def get_height(self, *, stream_index: Optional[int] = None) -> int:
        if stream_index is None:
            stream_index = self.default_stream_index

        return self.stream_infos[stream_index].height

    def get_num_color_channels(self, *, stream_index: Optional[int] = None) -> int:
        if stream_index is None:
            stream_index = self.default_stream_index

        return self.stream_infos[stream_index].num_color_channels

    def get_empty_chw_tensor(self, *, stream_index: int) -> torch.Tensor:
        return torch.empty(
            [
                0,
                self.get_num_color_channels(stream_index=stream_index),
                self.get_height(stream_index=stream_index),
                self.get_width(stream_index=stream_index),
            ],
            dtype=torch.uint8,
        )


NASA_VIDEO = TestVideo(
    filename="nasa_13013.mp4",
    default_stream_index=3,
    # This metadata is extracted manually.
    #  for stream index 0:
    #    $ ffprobe -v error -hide_banner -select_streams v:0 -show_frames -of json test/resources/nasa_13013.mp4 > out.json
    #
    #  for stream index 3:
    #    $ ffprobe -v error -hide_banner -select_streams v:1 -show_frames -of json test/resources/nasa_13013.mp4 > out.json
    #
    # Note that we are using the absolute stream index in the file. But ffprobe uses a relative stream
    # for that media type.
    stream_infos={
        0: TestVideoStreamInfo(width=320, height=180, num_color_channels=3),
        3: TestVideoStreamInfo(width=480, height=270, num_color_channels=3),
    },
    frames={
        0: {
            0: TestFrameInfo(pts_seconds=0.0, duration_seconds=0.040000),
            1: TestFrameInfo(pts_seconds=0.040000, duration_seconds=0.040000),
            2: TestFrameInfo(pts_seconds=0.080000, duration_seconds=0.040000),
            3: TestFrameInfo(pts_seconds=0.120000, duration_seconds=0.040000),
            4: TestFrameInfo(pts_seconds=0.160000, duration_seconds=0.040000),
            5: TestFrameInfo(pts_seconds=0.200000, duration_seconds=0.040000),
            6: TestFrameInfo(pts_seconds=0.240000, duration_seconds=0.040000),
            7: TestFrameInfo(pts_seconds=0.280000, duration_seconds=0.040000),
            8: TestFrameInfo(pts_seconds=0.320000, duration_seconds=0.040000),
            9: TestFrameInfo(pts_seconds=0.360000, duration_seconds=0.040000),
            10: TestFrameInfo(pts_seconds=0.400000, duration_seconds=0.040000),
        },
        3: {
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
            25: TestFrameInfo(pts_seconds=0.8342, duration_seconds=0.033367),
            35: TestFrameInfo(pts_seconds=1.1678, duration_seconds=0.033367),
        },
    },
)

# When we start actually decoding audio-only files, we'll probably need to define
# a TestAudio class with audio specific values. Until then, we only need a filename.
NASA_AUDIO = TestContainerFile(
    filename="nasa_13013.mp4.audio.mp3", default_stream_index=0, frames={}
)

H265_VIDEO = TestVideo(
    filename="h265_video.mp4",
    default_stream_index=0,
    # This metadata is extracted manually.
    #  $ ffprobe -v error -hide_banner -select_streams v:0 -show_frames -of json test/resources/h265_video.mp4 > out.json
    stream_infos={
        0: TestVideoStreamInfo(width=128, height=128, num_color_channels=3),
    },
    frames={
        0: {
            6: TestFrameInfo(pts_seconds=0.6, duration_seconds=0.1),
        },
    },
)

AV1_VIDEO = TestVideo(
    filename="av1_video.mkv",
    default_stream_index=0,
    # This metadata is extracted manually.
    #  $ ffprobe -v error -hide_banner -select_streams v:0 -show_frames -of json test/resources/av1_video.mkv > out.json
    stream_infos={
        0: TestVideoStreamInfo(width=640, height=360, num_color_channels=3),
    },
    frames={
        0: {
            10: TestFrameInfo(pts_seconds=0.400000, duration_seconds=0.040000),
        },
    },
)
