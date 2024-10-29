# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import json
import pathlib
from dataclasses import dataclass
from typing import List, Optional, Union

import torch

from torchcodec.decoders._core.video_decoder_ops import (
    _get_container_json_metadata,
    _get_stream_json_metadata,
    create_from_file,
)


@dataclass
class VideoStreamMetadata:
    """Metadata of a single video stream."""

    duration_seconds_from_header: Optional[float]
    """Duration of the stream, in seconds, obtained from the header (float or
    None). This could be inaccurate."""
    bit_rate: Optional[float]
    """Bit rate of the stream, in seconds (float or None)."""
    num_frames_from_header: Optional[int]
    """Number of frames, from the stream's metadata. This is potentially
    inaccurate. We recommend using the ``num_frames`` attribute instead.
    (int or None)."""
    num_frames_from_content: Optional[int]
    """Number of frames computed by TorchCodec by scanning the stream's
    content (the scan doesn't involve decoding). This is more accurate
    than ``num_frames_from_header``. We recommend using the
    ``num_frames`` attribute instead. (int or None)."""
    begin_stream_seconds: Optional[float]
    """Beginning of the stream, in seconds (float or None).
    Conceptually, this corresponds to the first frame's :term:`pts`. It is
    computed as min(frame.pts) across all frames in the stream. Usually, this is
    equal to 0."""
    end_stream_seconds: Optional[float]
    """End of the stream, in seconds (float or None).
    Conceptually, this corresponds to last_frame.pts + last_frame.duration. It
    is computed as max(frame.pts + frame.duration) across all frames in the
    stream. Note that no frame is played at this time value, so calling
    :meth:`~torchcodec.decoders.VideoDecoder.get_frame_played_at` with
    this value would result in an error. Retrieving the last frame is best done
    by simply indexing the :class:`~torchcodec.decoders.VideoDecoder`
    object with ``[-1]``.
    """
    codec: Optional[str]
    """Codec (str or None)."""
    width: Optional[int]
    """Width of the frames (int or None)."""
    height: Optional[int]
    """Height of the frames (int or None)."""
    average_fps_from_header: Optional[float]
    """Averate fps of the stream, obtained from the header (float or None).
    We recommend using the ``average_fps`` attribute instead."""
    stream_index: int
    """Index of the stream within the video (int)."""

    @property
    def num_frames(self) -> Optional[int]:
        """Number of frames in the stream. This corresponds to
        ``num_frames_from_content`` if a :term:`scan` was made, otherwise it
        corresponds to ``num_frames_from_header``.
        """
        if self.num_frames_from_content is not None:
            return self.num_frames_from_content
        else:
            return self.num_frames_from_header

    @property
    def duration_seconds(self) -> Optional[float]:
        """Duration of the stream in seconds. We try to calculate the duration
        from the actual frames if a :term:`scan` was performed. Otherwise we
        fall back to ``duration_seconds_from_header``.
        """
        if self.end_stream_seconds is None or self.begin_stream_seconds is None:
            return self.duration_seconds_from_header
        return self.end_stream_seconds - self.begin_stream_seconds

    @property
    def average_fps(self) -> Optional[float]:
        """Average fps of the stream. If a :term:`scan` was perfomed, this is
        computed from the number of frames and the duration of the stream.
        Otherwise we fall back to ``average_fps_from_header``.
        """
        if (
            self.end_stream_seconds is None
            or self.begin_stream_seconds is None
            or self.num_frames is None
        ):
            return self.average_fps_from_header
        return self.num_frames / (self.end_stream_seconds - self.begin_stream_seconds)

    def __repr__(self):
        # Overridden because properites are not printed by default.
        s = self.__class__.__name__ + ":\n"
        spaces = "  "
        s += f"{spaces}num_frames: {self.num_frames}\n"
        s += f"{spaces}duration_seconds: {self.duration_seconds}\n"
        s += f"{spaces}average_fps: {self.average_fps}\n"
        for field in dataclasses.fields(self):
            s += f"{spaces}{field.name}: {getattr(self, field.name)}\n"
        return s


@dataclass
class VideoMetadata:
    duration_seconds_from_header: Optional[float]
    bit_rate_from_header: Optional[float]
    best_video_stream_index: Optional[int]
    best_audio_stream_index: Optional[int]

    streams: List[VideoStreamMetadata]

    @property
    def duration_seconds(self) -> Optional[float]:
        raise NotImplementedError("Decide on logic and implement this!")

    @property
    def bit_rate(self) -> Optional[float]:
        raise NotImplementedError("Decide on logic and implement this!")

    @property
    def best_video_stream(self) -> VideoStreamMetadata:
        if self.best_video_stream_index is None:
            raise ValueError("The best video stream is unknown.")
        return self.streams[self.best_video_stream_index]


def get_video_metadata(decoder: torch.Tensor) -> VideoMetadata:
    """Return video metadata from a video decoder.

    The accuracy of the metadata and the availability of some returned fields
    depends on whether a full scan was performed by the decoder.
    """

    container_dict = json.loads(_get_container_json_metadata(decoder))
    streams_metadata = []
    for stream_index in range(container_dict["numStreams"]):
        stream_dict = json.loads(_get_stream_json_metadata(decoder, stream_index))
        streams_metadata.append(
            VideoStreamMetadata(
                duration_seconds_from_header=stream_dict.get("durationSeconds"),
                bit_rate=stream_dict.get("bitRate"),
                num_frames_from_header=stream_dict.get("numFrames"),
                num_frames_from_content=stream_dict.get("numFramesFromScan"),
                begin_stream_seconds=stream_dict.get("minPtsSecondsFromScan"),
                end_stream_seconds=stream_dict.get("maxPtsSecondsFromScan"),
                codec=stream_dict.get("codec"),
                width=stream_dict.get("width"),
                height=stream_dict.get("height"),
                average_fps_from_header=stream_dict.get("averageFps"),
                stream_index=stream_index,
            )
        )

    return VideoMetadata(
        duration_seconds_from_header=container_dict.get("durationSeconds"),
        bit_rate_from_header=container_dict.get("bitRate"),
        best_video_stream_index=container_dict.get("bestVideoStreamIndex"),
        best_audio_stream_index=container_dict.get("bestAudioStreamIndex"),
        streams=streams_metadata,
    )


def get_video_metadata_from_header(filename: Union[str, pathlib.Path]) -> VideoMetadata:
    return get_video_metadata(create_from_file(str(filename)))
