import importlib
import json
import os
import pathlib
import subprocess
import sys

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
import pytest

import torch

from torchcodec._core import get_ffmpeg_library_versions


# Decorator for skipping CUDA tests when CUDA isn't available. The tests are
# effectively marked to be skipped in pytest_collection_modifyitems() of
# conftest.py
def needs_cuda(test_item):
    return pytest.mark.needs_cuda(test_item)


# Decorator for skipping XPU tests when XPU isn't available. The tests are
# effectively marked to be skipped in pytest_collection_modifyitems() of
# conftest.py
def needs_xpu(test_item):
    return pytest.mark.needs_xpu(test_item)


def cpu_and_accelerators():
    return (
        "cpu",
        pytest.param("cuda", marks=pytest.mark.needs_cuda),
        pytest.param("xpu", marks=pytest.mark.needs_xpu),
    )


def get_ffmpeg_major_version():
    ffmpeg_version = get_ffmpeg_library_versions()["ffmpeg_version"]
    if ffmpeg_version.startswith("n"):
        ffmpeg_version = ffmpeg_version.removeprefix("n")
    return int(ffmpeg_version.split(".")[0])


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
        elif args[0].device.type == "xpu":
            if not torch.allclose(*args, atol=0, rtol=0):
                from torcheval.metrics import PeakSignalNoiseRatio

                metric = PeakSignalNoiseRatio()
                metric.update(args[0], args[1])
                assert metric.compute() >= 40
        else:
            torch.testing.assert_close(*args, **kwargs, atol=0, rtol=0)
    else:
        torch.testing.assert_close(*args, **kwargs, atol=3, rtol=0)


# Asserts that at least `percentage`% of the values are within the absolute tolerance.
# Percentage is expected in [0, 100] (actually, [60, 100])
def assert_tensor_close_on_at_least(
    actual_tensor, ref_tensor, *, percentage, atol, **kwargs
):
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


@dataclass
class TestFrameInfo:
    pts_seconds: float
    duration_seconds: float


@dataclass
class TestVideoStreamInfo:
    width: int
    height: int
    num_color_channels: int


@dataclass
class TestAudioStreamInfo:
    sample_rate: int
    num_channels: int
    duration_seconds: float
    num_frames: int
    sample_format: str


@dataclass
class TestContainerFile:
    __test__ = False  # prevents pytest from thinking this is a test class

    filename: str

    default_stream_index: int
    stream_infos: Dict[int, Union[TestVideoStreamInfo, TestAudioStreamInfo]]
    frames: Dict[int, Dict[int, TestFrameInfo]]
    _custom_frame_mappings_data: Dict[
        int, Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ] = field(default_factory=dict)

    def __post_init__(self):
        # We load the .frames attribute from the checked-in json files, if needed.
        # These frame info files are dumped with ffprobe, e.g.:
        # ```
        # ffprobe -v error -hide_banner -select_streams v:1 -show_frames -of json test/resources/nasa_13013.mp4 | jq '[.frames[] | {duration_time, pts_time}]'
        # ```
        # This will output the metadata for the frames of the second video
        # stream (v:1). First audio stream would be a:0.
        # Note that we are using the absolute stream index in the file. But
        # ffprobe uses a relative stream for that media type.
        for stream_index in self.stream_infos:
            if stream_index in self.frames:
                # .frames may be manually set: for some streams, we don't need
                # the info for all frames. We don't need to load anything in
                # this case
                continue

            frames_info_path = _get_file_path(
                f"{self.filename}.stream{stream_index}.all_frames_info.json"
            )

            if not frames_info_path.exists():
                raise ValueError(
                    f"Couldn't find {frames_info_path} for {self.filename}. "
                    "You need to submit this file, or specify the `frames` field manually."
                )

            with open(frames_info_path, "r") as f:
                frames_info = json.loads(f.read())
            self.frames[stream_index] = {
                frame_index: TestFrameInfo(
                    pts_seconds=float(frame_info["pts_time"]),
                    duration_seconds=float(frame_info["duration_time"]),
                )
                for frame_index, frame_info in enumerate(frames_info)
            }

    @property
    def path(self) -> pathlib.Path:
        return _get_file_path(self.filename)

    def to_tensor(self) -> torch.Tensor:
        arr = np.fromfile(self.path, dtype=np.uint8)
        return torch.from_numpy(arr)

    def get_frame_data_by_index(
        self, idx: int, *, stream_index: Optional[int] = None
    ) -> torch.Tensor:
        raise NotImplementedError("Override in child classes")

    def get_frame_data_by_range(
        self,
        start: int,
        stop: int,
        step: int = 1,
        *,
        stream_index: Optional[int] = None,
    ) -> torch.Tensor:
        raise NotImplementedError("Override in child classes")

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

    # This function is used to get the frame mappings for the custom_frame_mappings seek mode.
    def get_custom_frame_mappings(
        self, stream_index: Optional[int] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if stream_index is None:
            stream_index = self.default_stream_index
        if self._custom_frame_mappings_data.get(stream_index) is None:
            self.generate_custom_frame_mappings(stream_index)
        return self._custom_frame_mappings_data[stream_index]

    def generate_custom_frame_mappings(self, stream_index: int) -> None:
        result = json.loads(
            subprocess.run(
                [
                    "ffprobe",
                    "-i",
                    f"{self.path}",
                    "-select_streams",
                    f"{stream_index}",
                    "-show_frames",
                    "-of",
                    "json",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
        all_frames = torch.tensor([float(frame["pts"]) for frame in result["frames"]])
        is_key_frame = torch.tensor([frame["key_frame"] for frame in result["frames"]])
        duration = torch.tensor(
            [float(frame["duration"]) for frame in result["frames"]]
        )
        assert (
            len(all_frames) == len(is_key_frame) == len(duration)
        ), "Mismatched lengths in frame index data"
        self._custom_frame_mappings_data[stream_index] = (
            all_frames,
            is_key_frame,
            duration,
        )

    @property
    def empty_pts_seconds(self) -> torch.Tensor:
        return torch.empty([0], dtype=torch.float64)

    @property
    def empty_duration_seconds(self) -> torch.Tensor:
        return torch.empty([0], dtype=torch.float64)


@dataclass
class TestVideo(TestContainerFile):
    """Base class for the *video* streams of a video container"""

    def get_frame_data_by_index(
        self, idx: int, *, stream_index: Optional[int] = None
    ) -> torch.Tensor:
        if stream_index is None:
            stream_index = self.default_stream_index

        file_path = _get_file_path(
            f"{self.filename}.stream{stream_index}.frame{idx:06d}.pt"
        )
        return torch.load(file_path, weights_only=True).permute(2, 0, 1)

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
    stream_infos={
        0: TestVideoStreamInfo(width=320, height=180, num_color_channels=3),
        3: TestVideoStreamInfo(width=480, height=270, num_color_channels=3),
    },
    frames={},  # Automatically loaded from json file
)

# Video generated with:
# ffmpeg -f lavfi -i testsrc2=duration=1:size=200x200:rate=30 -c:v libx265 -pix_fmt yuv420p10le -preset fast -crf 23 h265_10bits.mp4
H265_10BITS = TestVideo(
    filename="h265_10bits.mp4",
    default_stream_index=0,
    stream_infos={
        0: TestVideoStreamInfo(width=200, height=200, num_color_channels=3),
    },
    frames={0: {}},  # Not needed yet
)

# Video generated with:
# peg -f lavfi -i testsrc2=duration=1:size=200x200:rate=30 -c:v libx264 -pix_fmt yuv420p10le -preset fast -crf 23 h264_10bits.mp4
H264_10BITS = TestVideo(
    filename="h264_10bits.mp4",
    default_stream_index=0,
    stream_infos={
        0: TestVideoStreamInfo(width=200, height=200, num_color_channels=3),
    },
    frames={0: {}},  # Not needed yet
)


@dataclass
class TestAudio(TestContainerFile):
    """Base class for the *audio* streams of a container (potentially a video),
    or a pure audio file"""

    stream_infos: Dict[int, TestAudioStreamInfo]
    # stream_index -> list of 2D frame tensors of shape (num_channels, num_samples_in_that_frame)
    # num_samples_in_that_frame isn't necessarily constant for a given stream.
    _reference_frames: Dict[int, List[torch.Tensor]] = field(default_factory=dict)

    # Storing each individual frame is too expensive for audio, because there's
    # a massive overhead in the binary format saved by pytorch. Saving all the
    # frames in a single file uses 1.6MB while saving all frames in individual
    # files uses 302MB (yes).
    # So we store the reference frames in a single file, and load/cache those
    # when the TestAudio instance is created.
    def __post_init__(self):
        super().__post_init__()
        for stream_index in self.stream_infos:
            frames_data_path = _get_file_path(
                f"{self.filename}.stream{stream_index}.all_frames.pt"
            )

            if frames_data_path.exists():
                # To ease development, we allow for the reference frames not to
                # exist. It means the asset cannot be used to check validity of
                # decoded frames.
                self._reference_frames[stream_index] = torch.load(
                    frames_data_path, weights_only=True
                )

    def get_frame_data_by_index(
        self, idx: int, *, stream_index: Optional[int] = None
    ) -> torch.Tensor:
        if stream_index is None:
            stream_index = self.default_stream_index

        return self._reference_frames[stream_index][idx]

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
        return torch.cat(tensors, dim=-1)

    def get_frame_index(
        self, *, pts_seconds: float, stream_index: Optional[int] = None
    ) -> int:
        if stream_index is None:
            stream_index = self.default_stream_index

        if pts_seconds <= self.frames[stream_index][0].pts_seconds:
            # Special case for e.g. NASA_AUDIO_MP3 whose first frame's pts is
            # 0.13~, not 0.
            return 0
        try:
            # Could use bisect() to maek this faster if needed
            return next(
                frame_index
                for (frame_index, frame_info) in self.frames[stream_index].items()
                if frame_info.pts_seconds
                <= pts_seconds
                < frame_info.pts_seconds + frame_info.duration_seconds
            )
        except StopIteration:
            return len(self.frames[stream_index]) - 1

    @property
    def sample_rate(self) -> int:
        return self.stream_infos[self.default_stream_index].sample_rate

    @property
    def num_channels(self) -> int:
        return self.stream_infos[self.default_stream_index].num_channels

    @property
    def duration_seconds(self) -> float:
        return self.stream_infos[self.default_stream_index].duration_seconds

    @property
    def num_frames(self) -> int:
        return self.stream_infos[self.default_stream_index].num_frames

    @property
    def sample_format(self) -> str:
        return self.stream_infos[self.default_stream_index].sample_format


NASA_AUDIO_MP3 = TestAudio(
    filename="nasa_13013.mp4.audio.mp3",
    default_stream_index=0,
    frames={},  # Automatically loaded from json file
    stream_infos={
        0: TestAudioStreamInfo(
            sample_rate=8_000,
            num_channels=2,
            duration_seconds=13.248,
            num_frames=183,
            sample_format="fltp",
        )
    },
)

# This file is the same as NASA_AUDIO_MP3, with a sample rate of 44_100. It was generated with:
# ffmpeg -i test/resources/nasa_13013.mp4.audio.mp3 -ar 44100 test/resources/nasa_13013.mp4.audio_44100.mp3
NASA_AUDIO_MP3_44100 = TestAudio(
    filename="nasa_13013.mp4.audio_44100.mp3",
    default_stream_index=0,
    frames={},  # Automatically loaded from json file
    stream_infos={
        0: TestAudioStreamInfo(
            sample_rate=44_100,
            num_channels=2,
            duration_seconds=13.09,
            num_frames=501,
            sample_format="fltp",
        )
    },
)

NASA_AUDIO = TestAudio(
    filename="nasa_13013.mp4",
    default_stream_index=4,
    frames={},  # Automatically loaded from json file
    stream_infos={
        4: TestAudioStreamInfo(
            sample_rate=16_000,
            num_channels=2,
            duration_seconds=13.056,
            num_frames=204,
            sample_format="fltp",
        )
    },
)

# Note that the file itself is s32 sample format, but the reference frames are
# stored as fltp. We can add the s32 original reference frames once we support
# decoding to non-fltp format, but for now we don't need to.
SINE_MONO_S32 = TestAudio(
    filename="sine_mono_s32.wav",
    default_stream_index=0,
    frames={},  # Automatically loaded from json file
    stream_infos={
        0: TestAudioStreamInfo(
            sample_rate=16_000,
            num_channels=1,
            duration_seconds=4,
            num_frames=63,
            sample_format="s32",
        )
    },
)

# This file is an upsampled version of SINE_MONO_S32, generated with:
# ffmpeg -i test/resources/sine_mono_s32.wav -ar 44100 -c:a pcm_s32le test/resources/sine_mono_s32_44100.wav
SINE_MONO_S32_44100 = TestAudio(
    filename="sine_mono_s32_44100.wav",
    default_stream_index=0,
    frames={},  # Automatically loaded from json file
    stream_infos={
        0: TestAudioStreamInfo(
            sample_rate=44_100,
            num_channels=1,
            duration_seconds=4,
            num_frames=173,
            sample_format="s32",
        )
    },
)

# This file is a downsampled version of SINE_MONO_S32, generated with:
# ffmpeg -i test/resources/sine_mono_s32.wav -ar 8000 -c:a pcm_s32le test/resources/sine_mono_s32_8000.wav
SINE_MONO_S32_8000 = TestAudio(
    filename="sine_mono_s32_8000.wav",
    default_stream_index=0,
    frames={},  # Automatically loaded from json file
    stream_infos={
        0: TestAudioStreamInfo(
            sample_rate=8000,
            num_channels=1,
            duration_seconds=4,
            num_frames=32,
            sample_format="s32",
        )
    },
)

# Same sample rate as SINE_MONO_S32, but encoded as s16 instead of s32. Generated with:
# ffmpeg -i test/resources/sine_mono_s32.wav -ar 16000 -c:a pcm_s16le test/resources/sine_mono_s16.wav
SINE_MONO_S16 = TestAudio(
    filename="sine_mono_s16.wav",
    default_stream_index=0,
    frames={},  # Automatically loaded from json file
    stream_infos={
        0: TestAudioStreamInfo(
            sample_rate=16_000,
            num_channels=1,
            duration_seconds=4,
            num_frames=63,
            sample_format="s16",
        )
    },
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
