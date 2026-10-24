# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import io
import json
import numbers
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
from torch import device as torch_device, nn, Tensor
from torchcodec import _core as core, Frame, FrameBatch
from torchcodec._core._decoder_utils import create_video_decoder
from torchcodec._logging import _LG
from torchcodec.decoders._decoder_utils import (
    _get_cuda_backend,
    convert_device_to_str,
    convert_output_dtype_to_str,
)
from torchcodec.transforms import DecoderTransform


@dataclass
class CpuFallbackStatus:
    """Information about CPU fallback status.

    This class tracks whether the decoder fell back to CPU decoding.
    Users should not instantiate this class directly; instead, access it
    via the :attr:`VideoDecoder.cpu_fallback` attribute.

    Usage:

    - Use ``str(cpu_fallback_status)`` or ``print(cpu_fallback_status)`` to see the cpu fallback status
    - Use ``if cpu_fallback_status:`` to check if any fallback occurred
    """

    status_known: bool = False
    """Whether the fallback status has been determined.
    For the NVDEC CUDA backend (the default; see
    :func:`~torchcodec.decoders.set_cuda_backend`), this is always ``True``
    immediately after decoder creation. For the FFmpeg CUDA backend, this
    becomes ``True`` after decoding the first frame."""
    _nvcuvid_unavailable: bool = field(default=False, init=False)
    _video_not_supported: bool = field(default=False, init=False)
    _is_fallback: bool = field(default=False, init=False)
    _backend: str = field(default="", init=False)

    def __bool__(self):
        """Returns True if fallback occurred."""
        return self.status_known and self._is_fallback

    def __str__(self):
        """Returns a human-readable string representation of the cpu fallback status."""
        if not self.status_known:
            return f"[{self._backend}] Fallback status: Unknown"

        reasons = []
        if self._nvcuvid_unavailable:
            reasons.append("NVcuvid unavailable")
        elif self._video_not_supported:
            reasons.append("Video not supported")
        elif self._is_fallback:
            reasons.append("Unknown reason - try the 'nvdec' backend to know more!")

        if reasons:
            return (
                f"[{self._backend}] Fallback status: Falling back due to: "
                + ", ".join(reasons)
            )
        return f"[{self._backend}] Fallback status: No fallback required"


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
        num_ffmpeg_threads (int, optional): The number of threads to use for CPU decoding.
            This has no effect when using GPU decoding.
            Use 1 for single-threaded decoding which may be best if you are running multiple
            instances of ``VideoDecoder`` in parallel. Use a higher number for multi-threaded
            decoding which is best if you are running a single instance of ``VideoDecoder``.
            Passing 0 lets FFmpeg decide on the number of threads.
            Default: 1.
        device (str or torch.device, optional): The device to use for decoding.
            If ``None`` (default), uses the current default device.
            If you pass a CUDA device, decoding uses the ``"nvdec"`` backend by
            default. See :func:`~torchcodec.decoders.set_cuda_backend` to switch
            to the ``"ffmpeg"`` CUDA backend.
        seek_mode (str, optional): Determines if frame access will be "exact" or
            "approximate". Exact guarantees that requesting frame i will always
            return frame i, but doing so requires an initial :term:`scan` of the
            file. Approximate is faster as it avoids scanning the file, but less
            accurate as it uses the file's metadata to calculate where i
            probably is. Default: "exact".
            Read more about this parameter in:
            :ref:`sphx_glr_generated_examples_decoding_approximate_mode.py`
        transforms (sequence of transform objects, optional): Sequence of transforms to be
            applied to the decoded frames by the decoder itself, in order. Accepts both
            :class:`~torchcodec.transforms.DecoderTransform` and
            :class:`~torchvision.transforms.v2.Transform`
            objects. Read more about this parameter in :ref:`sphx_glr_generated_examples_decoding_transforms.py`.
        output_dtype (torch.dtype or ``"auto"``, optional): The dtype of the
            output frames. Supported values are ``torch.uint8`` with values in
            [0, 255] (default), ``torch.float32`` with values in [0, 1], and
            ``"auto"``. When ``"auto"`` is specified, the output dtype is
            determined automatically based on the video content: uint8 for SDR
            content, float32 for HDR content.
            Read more about this parameter in:
            :ref:`sphx_glr_generated_examples_decoding_hdr_decoding.py`

            .. note::
                The features associated with this parameter are in beta stage
                and behavior may change in future versions. Specifically,
                detecting whether a video is SDR or HDR is difficult, so the
                "auto" heuristic is subject to change and improve.

        custom_frame_mappings (str, bytes, or file-like object, optional):
            Mapping of frames to their metadata, typically generated via ffprobe.
            This enables accurate frame seeking without requiring a full video scan.
            Do not set seek_mode when custom_frame_mappings is provided.
            Expected JSON format:

            .. code-block:: json

                {
                    "frames": [
                        {
                            "pts": 0,
                            "duration": 1001,
                            "key_frame": 1
                        }
                    ]
                }

            Alternative field names "pkt_pts" and "pkt_duration" are also supported.
            Read more about this parameter in:
            :ref:`sphx_glr_generated_examples_decoding_custom_frame_mappings.py`

    Attributes:
        metadata (VideoStreamMetadata): Metadata of the video stream.
        stream_index (int): The stream index that this decoder is retrieving frames from. If a
            stream index was provided at initialization, this is the same value. If it was left
            unspecified, this is the :term:`best stream`.
        cpu_fallback (CpuFallbackStatus): Information about whether the decoder fell back to CPU
            decoding. Use ``bool(cpu_fallback)`` to check if fallback occurred, or
            ``str(cpu_fallback)`` to get a human-readable status message. The status is only
            determined after at least one frame has been decoded.
    """

    def __init__(
        self,
        source: str | Path | io.RawIOBase | io.BufferedReader | bytes | Tensor,
        *,
        stream_index: int | None = None,
        dimension_order: Literal["NCHW", "NHWC"] = "NCHW",
        num_ffmpeg_threads: int = 1,
        device: str | torch_device | None = None,
        seek_mode: Literal["exact", "approximate"] = "exact",
        transforms: Sequence[DecoderTransform | nn.Module] | None = None,
        output_dtype: torch.dtype | Literal["auto"] = torch.uint8,
        custom_frame_mappings: (
            str | bytes | io.RawIOBase | io.BufferedReader | None
        ) = None,
    ):
        torch._C._log_api_usage_once("torchcodec.decoders.VideoDecoder")
        allowed_seek_modes = ("exact", "approximate")
        if seek_mode not in allowed_seek_modes:
            raise ValueError(
                f"Invalid seek mode ({seek_mode}). "
                f"Supported values are {', '.join(allowed_seek_modes)}."
            )

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

        allowed_dimension_orders = ("NCHW", "NHWC")
        if dimension_order not in allowed_dimension_orders:
            raise ValueError(
                f"Invalid dimension order ({dimension_order}). "
                f"Supported values are {', '.join(allowed_dimension_orders)}."
            )

        if num_ffmpeg_threads is None:
            raise ValueError(f"{num_ffmpeg_threads = } should be an int.")

        output_dtype = convert_output_dtype_to_str(output_dtype)

        device_variant = _get_cuda_backend()
        device = convert_device_to_str(device)

        if (
            device.startswith("cuda")
            and device_variant == "ffmpeg"
            and output_dtype != "uint8"
        ):
            raise ValueError(
                f"output_dtype={output_dtype} is not supported with the 'ffmpeg' "
                f"CUDA backend. Only torch.uint8 is supported. Use the default "
                f"'nvdec' CUDA backend for non-uint8 output dtypes."
            )

        (
            self._decoder,
            self.stream_index,
            self.metadata,
        ) = create_video_decoder(
            source=source,
            seek_mode=seek_mode,
            stream_index=stream_index,
            dimension_order=dimension_order,
            num_ffmpeg_threads=num_ffmpeg_threads,
            device=device,
            device_variant=device_variant,
            transforms=transforms,
            custom_frame_mappings=custom_frame_mappings_data,
            output_dtype=output_dtype,
        )

        assert self.metadata.begin_stream_seconds is not None  # mypy.
        assert self.metadata.end_stream_seconds is not None  # mypy.
        assert self.metadata.num_frames is not None  # mypy.

        _LG.debug(f"VideoDecoder created:\n{self.metadata}")

        self._begin_stream_seconds = self.metadata.begin_stream_seconds
        self._end_stream_seconds = self.metadata.end_stream_seconds
        self._num_frames = self.metadata.num_frames

        self._cpu_fallback = CpuFallbackStatus()
        if device.startswith("cuda"):
            if device_variant == "default":
                self._cpu_fallback._backend = "CUDA"
            else:
                self._cpu_fallback._backend = "FFmpeg CUDA"
        else:
            self._cpu_fallback._backend = "CPU"

    def __len__(self) -> int:
        return self._num_frames

    @property
    def cpu_fallback(self) -> CpuFallbackStatus:
        # We only query the CPU fallback info if status is unknown. That happens
        # either when:
        # - this @property has never been called before
        # - no frame has been decoded yet on the FFmpeg interface.
        # Note that for the NVDEC interface, we're able to know the fallback
        # status right when the VideoDecoder is instantiated, but the
        # status_known attribute is initialized to False.
        if not self._cpu_fallback.status_known:
            backend_details = core._get_backend_details(self._decoder)

            if "status unknown" not in backend_details:
                self._cpu_fallback.status_known = True

                if "CPU fallback" in backend_details:
                    self._cpu_fallback._is_fallback = True
                    if self._cpu_fallback._backend == "CUDA":
                        # Only the NVDEC interface can provide details.
                        # if it's not that nvcuvid is missing, it must be video-specific
                        if "NVCUVID not available" in backend_details:
                            self._cpu_fallback._nvcuvid_unavailable = True
                        else:
                            self._cpu_fallback._video_not_supported = True

        return self._cpu_fallback

    def _getitem_int(self, key: int) -> Tensor:
        assert isinstance(key, int)

        frame_data, *_ = core.get_frame_at_index(self._decoder, frame_index=key)
        return frame_data

    def _getitem_slice(self, key: slice) -> Tensor:
        assert isinstance(key, slice)

        start, stop, step = key.indices(len(self))
        _check_start_not_past_stop(start, stop, step)
        frame_data, *_ = core.get_frames_in_range(
            self._decoder,
            start=start,
            stop=stop,
            step=step,
        )
        return frame_data

    def __getitem__(self, key: numbers.Integral | slice) -> Tensor:
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

    def get_frames_at(self, indices: torch.Tensor | list[int]) -> FrameBatch:
        """Return frames at the given indices.

        Args:
            indices (torch.Tensor or list of int): The indices of the frames to retrieve.

        Returns:
            FrameBatch: The frames at the given indices.
        """

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
        _check_start_not_past_stop(start, stop, step)
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

    def get_frames_played_at(self, seconds: torch.Tensor | list[float]) -> FrameBatch:
        """Return frames played at the given timestamps in seconds.

        Args:
            seconds (torch.Tensor or list of float): The timestamps in seconds when the frames are played.

        Returns:
            FrameBatch: The frames that are played at ``seconds``.
        """

        data, pts_seconds, duration_seconds = core.get_frames_by_pts(
            self._decoder, timestamps=seconds
        )
        return FrameBatch(
            data=data,
            pts_seconds=pts_seconds,
            duration_seconds=duration_seconds,
        )

    def get_frames_played_in_range(
        self, start_seconds: float, stop_seconds: float, fps: float | None = None
    ) -> FrameBatch:
        """Returns multiple frames in the given range.

        Frames are in the half open range [start_seconds, stop_seconds). Each
        returned frame's :term:`pts`, in seconds, is inside of the half open
        range.

        Args:
            start_seconds (float): Time, in seconds, of the start of the range.
            stop_seconds (float): Time, in seconds, of the end of the range.
                As a half open range, the end is excluded.
            fps (float, optional): If specified, resample output to this frame
                rate by duplicating or dropping frames as necessary. If None
                (default), returns frames at the source video's frame rate.

        Returns:
            FrameBatch: The frames within the specified range.
        """
        if not start_seconds <= stop_seconds:
            raise ValueError(
                f"Invalid start seconds: {start_seconds}. "
                f"It must be less than or equal to stop seconds ({stop_seconds})."
            )
        if not self._begin_stream_seconds <= start_seconds < self._end_stream_seconds:
            raise ValueError(
                f"Invalid start seconds: {start_seconds}. "
                f"It must be greater than or equal to {self._begin_stream_seconds} "
                f"and less than {self._end_stream_seconds}."
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
            fps=fps,
        )
        return FrameBatch(*frames)

    def get_all_frames(self, fps: float | None = None) -> FrameBatch:
        """Returns all frames in the video.

        Args:
            fps (float, optional): If specified, resample output to this frame
                rate by duplicating or dropping frames as necessary. If None
                (default), returns frames at the source video's frame rate.

        Returns:
            FrameBatch: All frames in the video.
        """
        return self.get_frames_played_in_range(
            start_seconds=self._begin_stream_seconds,
            stop_seconds=self._end_stream_seconds,
            fps=fps,
        )


def _check_start_not_past_stop(start: int, stop: int, step: int) -> None:
    # start and stop are the indices returned by slice.indices(). Negative
    # steps are left to the core op, which rejects them.
    if step > 0 and start > stop:
        raise ValueError(
            f"Invalid start index: {start}. "
            f"It must be less than or equal to stop index ({stop})."
        )


def _read_custom_frame_mappings(
    custom_frame_mappings: str | bytes | io.RawIOBase | io.BufferedReader,
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

    all_frames = torch.tensor(
        [int(frame[pts_key]) for frame in input_data["frames"]], dtype=torch.int64
    )
    is_key_frame = torch.tensor(
        [int(frame["key_frame"]) for frame in input_data["frames"]], dtype=torch.bool
    )
    duration = torch.tensor(
        [int(frame[duration_key]) for frame in input_data["frames"]], dtype=torch.int64
    )
    if not (len(all_frames) == len(is_key_frame) == len(duration)):
        raise ValueError("Mismatched lengths in frame index data")
    return all_frames, is_key_frame, duration
