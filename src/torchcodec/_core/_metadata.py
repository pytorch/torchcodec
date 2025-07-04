# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import json
import pathlib
from dataclasses import dataclass
from fractions import Fraction
from typing import List, Optional, Union

import torch

from torchcodec._core.ops import (
    _get_container_json_metadata,
    _get_stream_json_metadata,
    create_from_file,
)


SPACES = "  "


@dataclass
class StreamMetadata:
    duration_seconds_from_header: Optional[float]
    """Duration of the stream, in seconds, obtained from the header (float or
    None). This could be inaccurate."""
    begin_stream_seconds_from_header: Optional[float]
    """Beginning of the stream, in seconds, obtained from the header (float or
    None). Usually, this is equal to 0."""
    bit_rate: Optional[float]
    """Bit rate of the stream, in seconds (float or None)."""
    codec: Optional[str]
    """Codec (str or None)."""
    stream_index: int
    """Index of the stream that this metadata refers to (int)."""

    def __repr__(self):
        s = self.__class__.__name__ + ":\n"
        for field in dataclasses.fields(self):
            s += f"{SPACES}{field.name}: {getattr(self, field.name)}\n"
        return s


@dataclass
class VideoStreamMetadata(StreamMetadata):
    """Metadata of a single video stream."""

    begin_stream_seconds_from_content: Optional[float]
    """Beginning of the stream, in seconds (float or None).
    Conceptually, this corresponds to the first frame's :term:`pts`. It is only
    computed when a :term:`scan` is done as min(frame.pts) across all frames in
    the stream. Usually, this is equal to 0."""
    end_stream_seconds_from_content: Optional[float]
    """End of the stream, in seconds (float or None).
    Conceptually, this corresponds to last_frame.pts + last_frame.duration. It
    is only computed when a :term:`scan` is done as max(frame.pts +
    frame.duration) across all frames in the stream. Note that no frame is
    played at this time value, so calling
    :meth:`~torchcodec.decoders.VideoDecoder.get_frame_played_at` with this
    value would result in an error. Retrieving the last frame is best done by
    simply indexing the :class:`~torchcodec.decoders.VideoDecoder` object with
    ``[-1]``.
    """
    width: Optional[int]
    """Width of the frames (int or None)."""
    height: Optional[int]
    """Height of the frames (int or None)."""
    num_frames_from_header: Optional[int]
    """Number of frames, from the stream's metadata. This is potentially
    inaccurate. We recommend using the ``num_frames`` attribute instead.
    (int or None)."""
    num_frames_from_content: Optional[int]
    """Number of frames computed by TorchCodec by scanning the stream's
    content (the scan doesn't involve decoding). This is more accurate
    than ``num_frames_from_header``. We recommend using the
    ``num_frames`` attribute instead. (int or None)."""
    average_fps_from_header: Optional[float]
    """Averate fps of the stream, obtained from the header (float or None).
    We recommend using the ``average_fps`` attribute instead."""
    pixel_aspect_ratio: Optional[Fraction]
    """Pixel Aspect Ratio (PAR), also known as Sample Aspect Ratio
    (SAR --- not to be confused with Storage Aspect Ratio, also SAR),
    is the ratio between the width and height of each pixel
    (``fractions.Fraction`` or None)."""

    @property
    def duration_seconds(self) -> Optional[float]:
        """Duration of the stream in seconds. We try to calculate the duration
        from the actual frames if a :term:`scan` was performed. Otherwise we
        fall back to ``duration_seconds_from_header``. If that value is also None,
        we  instead calculate the duration from ``num_frames_from_header`` and
        ``average_fps_from_header``.
        """
        if (
            self.end_stream_seconds_from_content is not None
            and self.begin_stream_seconds_from_content is not None
        ):
            return (
                self.end_stream_seconds_from_content
                - self.begin_stream_seconds_from_content
            )
        elif self.duration_seconds_from_header is not None:
            return self.duration_seconds_from_header
        elif (
            self.num_frames_from_header is not None
            and self.average_fps_from_header is not None
        ):
            return self.num_frames_from_header / self.average_fps_from_header
        else:
            return None

    @property
    def begin_stream_seconds(self) -> float:
        """Beginning of the stream, in seconds (float). Conceptually, this
        corresponds to the first frame's :term:`pts`. If
        ``begin_stream_seconds_from_content`` is not None, then it is returned.
        Otherwise, this value is 0.
        """
        if self.begin_stream_seconds_from_content is None:
            return 0
        else:
            return self.begin_stream_seconds_from_content

    @property
    def end_stream_seconds(self) -> Optional[float]:
        """End of the stream, in seconds (float or None).
        Conceptually, this corresponds to last_frame.pts + last_frame.duration.
        If ``end_stream_seconds_from_content`` is not None, then that value is
        returned. Otherwise, returns ``duration_seconds``.
        """
        if self.end_stream_seconds_from_content is None:
            return self.duration_seconds
        else:
            return self.end_stream_seconds_from_content

    @property
    def num_frames(self) -> Optional[int]:
        """Number of frames in the stream (int or None).
        This corresponds to ``num_frames_from_content`` if a :term:`scan` was made,
        otherwise it corresponds to ``num_frames_from_header``. If that value is also
        None, the number of frames is calculated from the duration and the average fps.
        """
        if self.num_frames_from_content is not None:
            return self.num_frames_from_content
        elif self.num_frames_from_header is not None:
            return self.num_frames_from_header
        elif (
            self.average_fps_from_header is not None
            and self.duration_seconds_from_header is not None
        ):
            return int(self.average_fps_from_header * self.duration_seconds_from_header)
        else:
            return None

    @property
    def average_fps(self) -> Optional[float]:
        """Average fps of the stream. If a :term:`scan` was perfomed, this is
        computed from the number of frames and the duration of the stream.
        Otherwise we fall back to ``average_fps_from_header``.
        """
        if (
            self.end_stream_seconds_from_content is None
            or self.begin_stream_seconds_from_content is None
            or self.num_frames is None
            # Should never happen, but prevents ZeroDivisionError:
            or self.end_stream_seconds_from_content
            == self.begin_stream_seconds_from_content
        ):
            return self.average_fps_from_header
        return self.num_frames / (
            self.end_stream_seconds_from_content
            - self.begin_stream_seconds_from_content
        )

    def __repr__(self):
        s = super().__repr__()
        s += f"{SPACES}duration_seconds: {self.duration_seconds}\n"
        s += f"{SPACES}begin_stream_seconds: {self.begin_stream_seconds}\n"
        s += f"{SPACES}end_stream_seconds: {self.end_stream_seconds}\n"
        s += f"{SPACES}num_frames: {self.num_frames}\n"
        s += f"{SPACES}average_fps: {self.average_fps}\n"
        return s


@dataclass
class AudioStreamMetadata(StreamMetadata):
    """Metadata of a single audio stream."""

    sample_rate: Optional[int]
    """The original sample rate."""
    num_channels: Optional[int]
    """The number of channels (1 for mono, 2 for stereo, etc.)"""
    sample_format: Optional[str]
    """The original sample format, as described by FFmpeg. E.g. 'fltp', 's32', etc."""

    def __repr__(self):
        return super().__repr__()


@dataclass
class ContainerMetadata:
    duration_seconds_from_header: Optional[float]
    bit_rate_from_header: Optional[float]
    best_video_stream_index: Optional[int]
    best_audio_stream_index: Optional[int]

    streams: List[StreamMetadata]

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
        metadata = self.streams[self.best_video_stream_index]
        assert isinstance(metadata, VideoStreamMetadata)  # mypy <3
        return metadata

    @property
    def best_audio_stream(self) -> AudioStreamMetadata:
        if self.best_audio_stream_index is None:
            raise ValueError("The best audio stream is unknown.")
        metadata = self.streams[self.best_audio_stream_index]
        assert isinstance(metadata, AudioStreamMetadata)  # mypy <3
        return metadata


def _get_optional_par_fraction(stream_dict):
    try:
        return Fraction(
            stream_dict["sampleAspectRatioNum"],
            stream_dict["sampleAspectRatioDen"],
        )
    except KeyError:
        return None


# TODO-AUDIO: This is user-facing. Should this just be `get_metadata`, without
# the "container" name in it? Same below.
def get_container_metadata(decoder: torch.Tensor) -> ContainerMetadata:
    """Return container metadata from a decoder.

    The accuracy of the metadata and the availability of some returned fields
    depends on whether a full scan was performed by the decoder.
    """

    container_dict = json.loads(_get_container_json_metadata(decoder))
    streams_metadata: List[StreamMetadata] = []
    for stream_index in range(container_dict["numStreams"]):
        stream_dict = json.loads(_get_stream_json_metadata(decoder, stream_index))
        common_meta = dict(
            duration_seconds_from_header=stream_dict.get("durationSecondsFromHeader"),
            bit_rate=stream_dict.get("bitRate"),
            begin_stream_seconds_from_header=stream_dict.get(
                "beginStreamSecondsFromHeader"
            ),
            codec=stream_dict.get("codec"),
            stream_index=stream_index,
        )
        if stream_dict["mediaType"] == "video":
            streams_metadata.append(
                VideoStreamMetadata(
                    begin_stream_seconds_from_content=stream_dict.get(
                        "beginStreamSecondsFromContent"
                    ),
                    end_stream_seconds_from_content=stream_dict.get(
                        "endStreamSecondsFromContent"
                    ),
                    width=stream_dict.get("width"),
                    height=stream_dict.get("height"),
                    num_frames_from_header=stream_dict.get("numFramesFromHeader"),
                    num_frames_from_content=stream_dict.get("numFramesFromContent"),
                    average_fps_from_header=stream_dict.get("averageFpsFromHeader"),
                    pixel_aspect_ratio=_get_optional_par_fraction(stream_dict),
                    **common_meta,
                )
            )
        elif stream_dict["mediaType"] == "audio":
            streams_metadata.append(
                AudioStreamMetadata(
                    sample_rate=stream_dict.get("sampleRate"),
                    num_channels=stream_dict.get("numChannels"),
                    sample_format=stream_dict.get("sampleFormat"),
                    **common_meta,
                )
            )
        else:
            # This is neither a video nor audio stream. Could be e.g. subtitles.
            # We still need to add a dummy entry so that len(streams_metadata)
            # is consistent with the number of streams.
            streams_metadata.append(StreamMetadata(**common_meta))

    return ContainerMetadata(
        duration_seconds_from_header=container_dict.get("durationSecondsFromHeader"),
        bit_rate_from_header=container_dict.get("bitRate"),
        best_video_stream_index=container_dict.get("bestVideoStreamIndex"),
        best_audio_stream_index=container_dict.get("bestAudioStreamIndex"),
        streams=streams_metadata,
    )


def get_container_metadata_from_header(
    filename: Union[str, pathlib.Path]
) -> ContainerMetadata:
    return get_container_metadata(
        create_from_file(str(filename), seek_mode="approximate")
    )
