# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Literal, Tuple, Union

from torch import Tensor

from torchcodec.decoders import _core as core


def _frame_repr(self):
    # Utility to replace Frame and FrameBatch __repr__ method. This prints the
    # shape of the .data tensor rather than printing the (potentially very long)
    # data tensor itself.
    s = self.__class__.__name__ + ":\n"
    spaces = "  "
    for field in dataclasses.fields(self):
        field_name = field.name
        field_val = getattr(self, field_name)
        if field_name == "data":
            field_name = "data (shape)"
            field_val = field_val.shape
        s += f"{spaces}{field_name}: {field_val}\n"
    return s


@dataclass
class Frame(Iterable):
    """A single video frame with associated metadata."""

    data: Tensor
    """The frame data as (3-D ``torch.Tensor``)."""
    pts_seconds: float
    """The :term:`pts` of the frame, in seconds (float)."""
    duration_seconds: float
    """The duration of the frame, in seconds (float)."""

    def __iter__(self) -> Iterator[Union[Tensor, float]]:
        for field in dataclasses.fields(self):
            yield getattr(self, field.name)

    def __repr__(self):
        return _frame_repr(self)


@dataclass
class FrameBatch(Iterable):
    """Multiple video frames with associated metadata."""

    data: Tensor
    """The frames data as (4-D ``torch.Tensor``)."""
    pts_seconds: Tensor
    """The :term:`pts` of the frame, in seconds (1-D ``torch.Tensor`` of floats)."""
    duration_seconds: Tensor
    """The duration of the frame, in seconds (1-D ``torch.Tensor`` of floats)."""

    def __iter__(self) -> Iterator[Union[Tensor, float]]:
        for field in dataclasses.fields(self):
            yield getattr(self, field.name)

    def __repr__(self):
        return _frame_repr(self)


_ERROR_REPORTING_INSTRUCTIONS = """
This should never happen. Please report an issue following the steps in
https://github.com/pytorch/torchcodec/issues/new?assignees=&labels=&projects=&template=bug-report.yml.
"""


class SimpleVideoDecoder:
    """A single-stream video decoder.

    If the video contains multiple video streams, the :term:`best stream` is
    used. This decoder always performs a :term:`scan` of the video.

    Args:
        source (str, ``Pathlib.path``, ``torch.Tensor``, or bytes): The source of the video.

            - If ``str`` or ``Pathlib.path``: a path to a local video file.
            - If ``bytes`` object or ``torch.Tensor``: the raw encoded video data.
        dimension_order(str, optional): The dimension order of the decoded frames.
            This can be either "NCHW" (default) or "NHWC", where N is the batch
            size, C is the number of channels, H is the height, and W is the
            width of the frames.

            .. note::

                Frames are natively decoded in NHWC format by the underlying
                FFmpeg implementation. Converting those into NCHW format is a
                cheap no-copy operation that allows these frames to be
                transformed using the `torchvision transforms
                <https://pytorch.org/vision/stable/transforms.html>`_.

    Attributes:
        metadata (VideoStreamMetadata): Metadata of the video stream.
    """

    def __init__(
        self,
        source: Union[str, Path, bytes, Tensor],
        dimension_order: Literal["NCHW", "NHWC"] = "NCHW",
    ):
        if isinstance(source, str):
            self._decoder = core.create_from_file(source)
        elif isinstance(source, Path):
            self._decoder = core.create_from_file(str(source))
        elif isinstance(source, bytes):
            self._decoder = core.create_from_bytes(source)
        elif isinstance(source, Tensor):
            self._decoder = core.create_from_tensor(source)
        else:
            raise TypeError(
                f"Unknown source type: {type(source)}. "
                "Supported types are str, Path, bytes and Tensor."
            )

        allowed_dimension_orders = ("NCHW", "NHWC")
        if dimension_order not in allowed_dimension_orders:
            raise ValueError(
                f"Invalid dimension order ({dimension_order}). "
                f"Supported values are {', '.join(allowed_dimension_orders)}."
            )

        core.scan_all_streams_to_update_metadata(self._decoder)
        core.add_video_stream(self._decoder, dimension_order=dimension_order)

        self.metadata, self._stream_index = _get_and_validate_stream_metadata(
            self._decoder
        )

        if self.metadata.num_frames_from_content is None:
            raise ValueError(
                "The number of frames is unknown. " + _ERROR_REPORTING_INSTRUCTIONS
            )
        self._num_frames = self.metadata.num_frames_from_content

        if self.metadata.begin_stream_seconds is None:
            raise ValueError(
                "The minimum pts value in seconds is unknown. "
                + _ERROR_REPORTING_INSTRUCTIONS
            )
        self._begin_stream_seconds = self.metadata.begin_stream_seconds

        if self.metadata.end_stream_seconds is None:
            raise ValueError(
                "The maximum pts value in seconds is unknown. "
                + _ERROR_REPORTING_INSTRUCTIONS
            )
        self._end_stream_seconds = self.metadata.end_stream_seconds

    def __len__(self) -> int:
        return self._num_frames

    def _getitem_int(self, key: int) -> Tensor:
        assert isinstance(key, int)

        if key < 0:
            key += self._num_frames
        if key >= self._num_frames or key < 0:
            raise IndexError(
                f"Index {key} is out of bounds; length is {self._num_frames}"
            )

        frame_data, *_ = core.get_frame_at_index(
            self._decoder, frame_index=key, stream_index=self._stream_index
        )
        return frame_data

    def _getitem_slice(self, key: slice) -> Tensor:
        assert isinstance(key, slice)

        start, stop, step = key.indices(len(self))
        frame_data, *_ = core.get_frames_in_range(
            self._decoder,
            stream_index=self._stream_index,
            start=start,
            stop=stop,
            step=step,
        )
        return frame_data

    def __getitem__(self, key: Union[int, slice]) -> Tensor:
        """Return frame or frames as tensors, at the given index or range.

        Args:
            key(int or slice): The index or range of frame(s) to retrieve.

        Returns:
            torch.Tensor: The frame or frames at the given index or range.
        """
        if isinstance(key, int):
            return self._getitem_int(key)
        elif isinstance(key, slice):
            return self._getitem_slice(key)

        raise TypeError(
            f"Unsupported key type: {type(key)}. Supported types are int and slice."
        )

    def get_frame_at(self, index: int) -> Frame:
        """Return a single frame at the given index.

        Args:
            index (int): The index of the frame to retrieve.

        Returns:
            Frame: The frame at the given index.
        """

        if not 0 <= index < self._num_frames:
            raise IndexError(
                f"Index {index} is out of bounds; must be in the range [0, {self._num_frames})."
            )
        data, pts_seconds, duration_seconds = core.get_frame_at_index(
            self._decoder, frame_index=index, stream_index=self._stream_index
        )
        return Frame(
            data=data,
            pts_seconds=pts_seconds.item(),
            duration_seconds=duration_seconds.item(),
        )

    def get_frames_at(self, start: int, stop: int, step: int = 1) -> FrameBatch:
        """Return multiple frames at the given index range.

        Frames are in [start, stop).

        Args:
            start (int): Index of the first frame to retrieve.
            stop (int): End of indexing range (exclusive, as per Python
                conventions).
            step (int, optional): Step size between frames. Default: 1.

        Returns:
            FrameBatch: The frames within the specified range.
        """
        if not 0 <= start < self._num_frames:
            raise IndexError(
                f"Start index {start} is out of bounds; must be in the range [0, {self._num_frames})."
            )
        if stop < start:
            raise IndexError(
                f"Stop index ({stop}) must not be less than the start index ({start})."
            )
        if not step > 0:
            raise IndexError(f"Step ({step}) must be greater than 0.")
        frames = core.get_frames_in_range(
            self._decoder,
            stream_index=self._stream_index,
            start=start,
            stop=stop,
            step=step,
        )
        return FrameBatch(*frames)

    def get_frame_displayed_at(self, seconds: float) -> Frame:
        """Return a single frame displayed at the given timestamp in seconds.

        Args:
            seconds (float): The time stamp in seconds when the frame is displayed.

        Returns:
            Frame: The frame that is displayed at ``seconds``.
        """
        if not self._begin_stream_seconds <= seconds < self._end_stream_seconds:
            raise IndexError(
                f"Invalid pts in seconds: {seconds}. "
                f"It must be greater than or equal to {self._begin_stream_seconds} "
                f"and less than {self._end_stream_seconds}."
            )
        data, pts_seconds, duration_seconds = core.get_frame_at_pts(
            self._decoder, seconds
        )
        return Frame(
            data=data,
            pts_seconds=pts_seconds.item(),
            duration_seconds=duration_seconds.item(),
        )

    def get_frames_displayed_at(
        self, start_seconds: float, stop_seconds: float
    ) -> FrameBatch:
        """Returns multiple frames in the given range.

        Frames are in the half open range [start_seconds, stop_seconds). Each
        returned frame's :term:`pts`, in seconds, is inside of the half open
        range.

        Args:
            start_seconds (float): Time, in seconds, of the start of the
                range.
            stop_seconds (float): Time, in seconds, of the end of the
                range. As a half open range, the end is excluded.

        Returns:
            FrameBatch: The frames within the specified range.
        """
        if not start_seconds <= stop_seconds:
            raise ValueError(
                f"Invalid start seconds: {start_seconds}. It must be less than or equal to stop seconds ({stop_seconds})."
            )
        if not self._begin_stream_seconds <= start_seconds < self._end_stream_seconds:
            raise ValueError(
                f"Invalid start seconds: {start_seconds}. "
                f"It must be greater than or equal to {self._begin_stream_seconds} "
                f"and less than or equal to {self._end_stream_seconds}."
            )
        if not stop_seconds <= self._end_stream_seconds:
            raise ValueError(
                f"Invalid stop seconds: {stop_seconds}. "
                f"It must be less than or equal to {self._end_stream_seconds}."
            )
        frames = core.get_frames_by_pts_in_range(
            self._decoder,
            stream_index=self._stream_index,
            start_seconds=start_seconds,
            stop_seconds=stop_seconds,
        )
        return FrameBatch(*frames)


def _get_and_validate_stream_metadata(
    decoder: Tensor,
) -> Tuple[core.VideoStreamMetadata, int]:
    video_metadata = core.get_video_metadata(decoder)

    best_stream_index = video_metadata.best_video_stream_index
    if best_stream_index is None:
        raise ValueError(
            "The best video stream is unknown. " + _ERROR_REPORTING_INSTRUCTIONS
        )

    best_stream_metadata = video_metadata.streams[best_stream_index]
    return (best_stream_metadata, best_stream_index)
