# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import json
import numbers
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import torch
from torch import device as torch_device, Tensor

from torchcodec import _core as core, Frame, FrameBatch
from torchcodec.decoders._decoder_utils import (
    create_decoder,
    ERROR_REPORTING_INSTRUCTIONS,
)


class VideoDecoder:
    """A single-stream video decoder.

    Args:
        source (str, ``Pathlib.path``, bytes, ``torch.Tensor`` or file-like object): The source of the video:

            - If ``str``: a local path or a URL to a video file.
            - If ``Pathlib.path``: a path to a local video file.
            - If ``bytes`` object or ``torch.Tensor``: the raw encoded video data.
            - If file-like object: we read video data from the object on demand. The object must
              expose the methods `read(self, size: int) -> bytes` and
              `seek(self, offset: int, whence: int) -> int`. Read more in:
              :ref:`sphx_glr_generated_examples_decoding_file_like.py`.
        stream_index (int, optional): Specifies which stream in the video to decode frames from.
            Note that this index is absolute across all media types. If left unspecified, then
            the :term:`best stream` is used.
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
        num_ffmpeg_threads (int, optional): The number of threads to use for decoding.
            Use 1 for single-threaded decoding which may be best if you are running multiple
            instances of ``VideoDecoder`` in parallel. Use a higher number for multi-threaded
            decoding which is best if you are running a single instance of ``VideoDecoder``.
            Passing 0 lets FFmpeg decide on the number of threads.
            Default: 1.
        device (str or torch.device, optional): The device to use for decoding. Default: "cpu".
        seek_mode (str, optional): Determines if frame access will be "exact" or
            "approximate". Exact guarantees that requesting frame i will always
            return frame i, but doing so requires an initial :term:`scan` of the
            file. Approximate is faster as it avoids scanning the file, but less
            accurate as it uses the file's metadata to calculate where i
            probably is. Default: "exact".
            Read more about this parameter in:
            :ref:`sphx_glr_generated_examples_decoding_approximate_mode.py`

    Attributes:
        metadata (VideoStreamMetadata): Metadata of the video stream.
        stream_index (int): The stream index that this decoder is retrieving frames from. If a
            stream index was provided at initialization, this is the same value. If it was left
            unspecified, this is the :term:`best stream`.
    """

    def __init__(
        self,
        source: Union[str, Path, io.RawIOBase, io.BufferedReader, bytes, Tensor],
        *,
        stream_index: Optional[int] = None,
        dimension_order: Literal["NCHW", "NHWC"] = "NCHW",
        num_ffmpeg_threads: int = 1,
        device: Optional[Union[str, torch_device]] = "cpu",
        seek_mode: Literal["exact", "approximate"] = "exact",
    ):
        torch._C._log_api_usage_once("torchcodec.decoders.VideoDecoder")
        allowed_seek_modes = ("exact", "approximate")
        if seek_mode not in allowed_seek_modes:
            raise ValueError(
                f"Invalid seek mode ({seek_mode}). "
                f"Supported values are {', '.join(allowed_seek_modes)}."
            )

        custom_frame_mappings = None
        # Validate seek_mode and custom_frame_mappings are not mismatched
        if custom_frame_mappings is not None and seek_mode == "approximate":
            raise ValueError(
                "custom_frame_mappings is incompatible with seek_mode='approximate'. "
                "Use seek_mode='custom_frame_mappings' or leave it unspecified to automatically use custom frame mappings."
            )

        # Auto-select custom_frame_mappings seek_mode and process data when mappings are provided
        custom_frame_mappings_data = None
        if custom_frame_mappings is not None:
            seek_mode = "custom_frame_mappings"  # type: ignore[assignment]
            custom_frame_mappings_data = _read_custom_frame_mappings(
                custom_frame_mappings
            )

        self._decoder = create_decoder(source=source, seek_mode=seek_mode)

        allowed_dimension_orders = ("NCHW", "NHWC")
        if dimension_order not in allowed_dimension_orders:
            raise ValueError(
                f"Invalid dimension order ({dimension_order}). "
                f"Supported values are {', '.join(allowed_dimension_orders)}."
            )

        if num_ffmpeg_threads is None:
            raise ValueError(f"{num_ffmpeg_threads = } should be an int.")

        if isinstance(device, torch_device):
            device = str(device)

        core.add_video_stream(
            self._decoder,
            stream_index=stream_index,
            dimension_order=dimension_order,
            num_threads=num_ffmpeg_threads,
            device=device,
            custom_frame_mappings=custom_frame_mappings_data,
        )

        (
            self.metadata,
            self.stream_index,
            self._begin_stream_seconds,
            self._end_stream_seconds,
            self._num_frames,
        ) = _get_and_validate_stream_metadata(
            decoder=self._decoder, stream_index=stream_index
        )

    def __len__(self) -> int:
        return self._num_frames

    def _getitem_int(self, key: int) -> Tensor:
        assert isinstance(key, int)

        frame_data, *_ = core.get_frame_at_index(self._decoder, frame_index=key)
        return frame_data

    def _getitem_slice(self, key: slice) -> Tensor:
        assert isinstance(key, slice)

        start, stop, step = key.indices(len(self))
        frame_data, *_ = core.get_frames_in_range(
            self._decoder,
            start=start,
            stop=stop,
            step=step,
        )
        return frame_data

    def __getitem__(self, key: Union[numbers.Integral, slice]) -> Tensor:
        """Return frame or frames as tensors, at the given index or range.

        .. note::

            If you need to decode multiple frames, we recommend using the batch
            methods instead, since they are faster:
            :meth:`~torchcodec.decoders.VideoDecoder.get_frames_at`,
            :meth:`~torchcodec.decoders.VideoDecoder.get_frames_in_range`,
            :meth:`~torchcodec.decoders.VideoDecoder.get_frames_played_at`, and
            :meth:`~torchcodec.decoders.VideoDecoder.get_frames_played_in_range`.

        Args:
            key(int or slice): The index or range of frame(s) to retrieve.

        Returns:
            torch.Tensor: The frame or frames at the given index or range.
        """
        if isinstance(key, numbers.Integral):
            return self._getitem_int(int(key))
        elif isinstance(key, slice):
            return self._getitem_slice(key)

        raise TypeError(
            f"Unsupported key type: {type(key)}. Supported types are int and slice."
        )

    def _get_key_frame_indices(self) -> list[int]:
        return core._get_key_frame_indices(self._decoder)

    def get_frame_at(self, index: int) -> Frame:
        """Return a single frame at the given index.

        .. note::

            If you need to decode multiple frames, we recommend using the batch
            methods instead, since they are faster:
            :meth:`~torchcodec.decoders.VideoDecoder.get_frames_at`,
            :meth:`~torchcodec.decoders.VideoDecoder.get_frames_in_range`,
            :meth:`~torchcodec.decoders.VideoDecoder.get_frames_played_at`,
            :meth:`~torchcodec.decoders.VideoDecoder.get_frames_played_in_range`.

        Args:
            index (int): The index of the frame to retrieve.

        Returns:
            Frame: The frame at the given index.
        """
        data, pts_seconds, duration_seconds = core.get_frame_at_index(
            self._decoder, frame_index=index
        )
        return Frame(
            data=data,
            pts_seconds=pts_seconds.item(),
            duration_seconds=duration_seconds.item(),
        )

    def get_frames_at(self, indices: list[int]) -> FrameBatch:
        """Return frames at the given indices.

        Args:
            indices (list of int): The indices of the frames to retrieve.

        Returns:
            FrameBatch: The frames at the given indices.
        """
        if isinstance(indices, torch.Tensor):
            # TODO we should avoid converting tensors to lists and just let the
            # core ops and C++ code natively accept tensors.  See
            # https://github.com/pytorch/torchcodec/issues/879
            indices = indices.to(torch.int).tolist()

        data, pts_seconds, duration_seconds = core.get_frames_at_indices(
            self._decoder, frame_indices=indices
        )
        return FrameBatch(
            data=data,
            pts_seconds=pts_seconds,
            duration_seconds=duration_seconds,
        )

    def get_frames_in_range(self, start: int, stop: int, step: int = 1) -> FrameBatch:
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
        # Adjust start / stop indices to enable indexing semantics, ex. [-10, 1000] returns the last 10 frames
        start, stop, step = slice(start, stop, step).indices(self._num_frames)
        frames = core.get_frames_in_range(
            self._decoder,
            start=start,
            stop=stop,
            step=step,
        )
        return FrameBatch(*frames)

    def get_frame_played_at(self, seconds: float) -> Frame:
        """Return a single frame played at the given timestamp in seconds.

        .. note::

            If you need to decode multiple frames, we recommend using the batch
            methods instead, since they are faster:
            :meth:`~torchcodec.decoders.VideoDecoder.get_frames_at`,
            :meth:`~torchcodec.decoders.VideoDecoder.get_frames_in_range`,
            :meth:`~torchcodec.decoders.VideoDecoder.get_frames_played_at`,
            :meth:`~torchcodec.decoders.VideoDecoder.get_frames_played_in_range`.

        Args:
            seconds (float): The time stamp in seconds when the frame is played.

        Returns:
            Frame: The frame that is played at ``seconds``.
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

    def get_frames_played_at(self, seconds: list[float]) -> FrameBatch:
        """Return frames played at the given timestamps in seconds.

        Args:
            seconds (list of float): The timestamps in seconds when the frames are played.

        Returns:
            FrameBatch: The frames that are played at ``seconds``.
        """
        if isinstance(seconds, torch.Tensor):
            # TODO we should avoid converting tensors to lists and just let the
            # core ops and C++ code natively accept tensors.  See
            # https://github.com/pytorch/torchcodec/issues/879
            seconds = seconds.to(torch.float).tolist()

        data, pts_seconds, duration_seconds = core.get_frames_by_pts(
            self._decoder, timestamps=seconds
        )
        return FrameBatch(
            data=data,
            pts_seconds=pts_seconds,
            duration_seconds=duration_seconds,
        )

    def get_frames_played_in_range(
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
            start_seconds=start_seconds,
            stop_seconds=stop_seconds,
        )
        return FrameBatch(*frames)


def _get_and_validate_stream_metadata(
    *,
    decoder: Tensor,
    stream_index: Optional[int] = None,
) -> Tuple[core._metadata.VideoStreamMetadata, int, float, float, int]:

    container_metadata = core.get_container_metadata(decoder)

    if stream_index is None:
        if (stream_index := container_metadata.best_video_stream_index) is None:
            raise ValueError(
                "The best video stream is unknown and there is no specified stream. "
                + ERROR_REPORTING_INSTRUCTIONS
            )

    metadata = container_metadata.streams[stream_index]
    assert isinstance(metadata, core._metadata.VideoStreamMetadata)  # mypy

    if metadata.begin_stream_seconds is None:
        raise ValueError(
            "The minimum pts value in seconds is unknown. "
            + ERROR_REPORTING_INSTRUCTIONS
        )
    begin_stream_seconds = metadata.begin_stream_seconds

    if metadata.end_stream_seconds is None:
        raise ValueError(
            "The maximum pts value in seconds is unknown. "
            + ERROR_REPORTING_INSTRUCTIONS
        )
    end_stream_seconds = metadata.end_stream_seconds

    if metadata.num_frames is None:
        raise ValueError(
            "The number of frames is unknown. " + ERROR_REPORTING_INSTRUCTIONS
        )
    num_frames = metadata.num_frames

    return (
        metadata,
        stream_index,
        begin_stream_seconds,
        end_stream_seconds,
        num_frames,
    )


def _read_custom_frame_mappings(
    custom_frame_mappings: Union[str, bytes, io.RawIOBase, io.BufferedReader]
) -> tuple[Tensor, Tensor, Tensor]:
    """Parse custom frame mappings from JSON data and extract frame metadata.

    Args:
        custom_frame_mappings: JSON data containing frame metadata, provided as:
            - A JSON string (str, bytes)
            - A file-like object with a read() method

    Returns:
        A tuple of three tensors:
        - all_frames (Tensor): Presentation timestamps (PTS) for each frame
        - is_key_frame (Tensor): Boolean tensor indicating which frames are key frames
        - duration (Tensor): Duration of each frame
    """
    try:
        input_data = (
            json.load(custom_frame_mappings)
            if hasattr(custom_frame_mappings, "read")
            else json.loads(custom_frame_mappings)
        )
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid custom frame mappings: {e}. It should be a valid JSON string or a file-like object."
        ) from e

    if not input_data or "frames" not in input_data:
        raise ValueError(
            "Invalid custom frame mappings. The input is empty or missing the required 'frames' key."
        )

    first_frame = input_data["frames"][0]
    pts_key = next((key for key in ("pts", "pkt_pts") if key in first_frame), None)
    duration_key = next(
        (key for key in ("duration", "pkt_duration") if key in first_frame), None
    )
    key_frame_present = "key_frame" in first_frame

    if not pts_key or not duration_key or not key_frame_present:
        raise ValueError(
            "Invalid custom frame mappings. The 'pts'/'pkt_pts', 'duration'/'pkt_duration', and 'key_frame' keys are required in the frame metadata."
        )

    frame_data = [
        (float(frame[pts_key]), frame["key_frame"], float(frame[duration_key]))
        for frame in input_data["frames"]
    ]
    all_frames, is_key_frame, duration = map(torch.tensor, zip(*frame_data))
    if not (len(all_frames) == len(is_key_frame) == len(duration)):
        raise ValueError("Mismatched lengths in frame index data")
    return all_frames, is_key_frame, duration
