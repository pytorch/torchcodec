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
from typing import Any

import torch
from torchcodec._core.ops import (
    _get_container_json_metadata,
    _get_stream_json_metadata,
    create_from_file,
)


SPACES = "  "


@dataclass
class StreamMetadata:
    """Metadata of a single stream, as reported by the container header.

    This is everything that is known about a stream without looking at its
    content. Streams that are neither video nor audio (subtitles, data) are
    described by this class and nothing more.
    """

    duration_seconds_from_header: float | None
    """Duration of the stream, in seconds, obtained from the header (float or
    None). This could be inaccurate."""
    begin_stream_seconds_from_header: float | None
    """Beginning of the stream, in seconds, obtained from the header (float or
    None). Usually, this is equal to 0."""
    bit_rate: float | None
    """Bit rate of the stream, in seconds (float or None)."""
    codec: str | None
    """Codec (str or None)."""
    stream_index: int
    """Index of the stream that this metadata refers to (int)."""

    def __repr__(self):
        s = self.__class__.__name__ + ":\n"
        for field in dataclasses.fields(self):
            s += f"{SPACES}{field.name}: {getattr(self, field.name)}\n"
        return s


@dataclass
class VideoStreamHeaderMetadata(StreamMetadata):
    """Metadata of a single video stream, as reported by the container header."""

    width: int | None
    """Width of the frames (int or None)."""
    height: int | None
    """Height of the frames (int or None)."""
    num_frames_from_header: int | None
    """Number of frames, from the stream's metadata. This is potentially
    inaccurate. We recommend using the ``num_frames`` attribute instead.
    (int or None)."""
    average_fps_from_header: float | None
    """Averate fps of the stream, obtained from the header (float or None).
    We recommend using the ``average_fps`` attribute instead."""
    pixel_aspect_ratio: Fraction | None
    """Pixel Aspect Ratio (PAR), also known as Sample Aspect Ratio
    (SAR --- not to be confused with Storage Aspect Ratio, also SAR),
    is the ratio between the width and height of each pixel
    (``fractions.Fraction`` or None)."""
    rotation: float | None
    """Rotation angle in degrees (counter-clockwise rounded to the nearest
    multiple of 90 degrees) from the display matrix metadata. This indicates
    how the video should be rotated for correct display. TorchCodec automatically
    applies this rotation during decoding, so the returned frames are in the
    correct orientation (float or None).

    .. note::

        The :attr:`~torchcodec.decoders.VideoStreamMetadata.width` and
        :attr:`~torchcodec.decoders.VideoStreamMetadata.height` attributes report
        the **post-rotation** dimensions, i.e., the dimensions of frames as they
        will be returned by TorchCodec's decoding methods. For videos with 90
        or -90 degree rotation, this means width and height are swapped
        compared to the raw encoded dimensions in the container.
    """
    color_primaries: str | None
    """Color primaries as reported by FFmpeg. E.g. ``"bt709"``, ``"bt2020"``."""
    color_space: str | None
    """Color space as reported by FFmpeg. E.g. ``"bt709"``,
    ``"bt2020nc"``."""
    color_transfer_characteristic: str | None
    """Color transfer characteristic as reported by FFmpeg
    E.g. ``"bt709"``, ``"smpte2084"`` (PQ), ``"arib-std-b67"`` (HLG)."""
    pixel_format: str | None
    """The source pixel format of the video as reported by FFmpeg.
    E.g. ``'yuv420p'``, ``'yuv444p'``, etc."""

    def __repr__(self):
        return super().__repr__()


@dataclass
class VideoStreamMetadata(VideoStreamHeaderMetadata):
    """Metadata of a single video stream."""

    begin_stream_seconds_from_content: float | None
    """Beginning of the stream, in seconds (float or None).
    Conceptually, this corresponds to the first frame's :term:`pts`. It is only
    computed when a :term:`scan` is done as min(frame.pts) across all frames in
    the stream. Usually, this is equal to 0."""
    end_stream_seconds_from_content: float | None
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
    num_frames_from_content: int | None
    """Number of frames computed by TorchCodec by scanning the stream's
    content (the scan doesn't involve decoding). This is more accurate
    than ``num_frames_from_header``. We recommend using the
    ``num_frames`` attribute instead. (int or None)."""

    # Computed fields (computed in C++ with fallback logic)
    duration_seconds: float | None
    """Duration of the stream in seconds. We try to calculate the duration
    from the actual frames if a :term:`scan` was performed. Otherwise we
    fall back to ``duration_seconds_from_header``. If that value is also None,
    we instead calculate the duration from ``num_frames_from_header`` and
    ``average_fps_from_header``. If all of those are unavailable, we fall back
    to the container-level ``duration_seconds_from_header``.
    """
    begin_stream_seconds: float | None
    """Beginning of the stream, in seconds (float). Conceptually, this
    corresponds to the first frame's :term:`pts`. If a :term:`scan` was performed
    and ``begin_stream_seconds_from_content`` is not None, then it is returned.
    Otherwise, this value is 0.
    """
    end_stream_seconds: float | None
    """End of the stream, in seconds (float or None).
    Conceptually, this corresponds to last_frame.pts + last_frame.duration.
    If :term:`scan` was performed and``end_stream_seconds_from_content`` is not None, then that value is
    returned. Otherwise, returns ``duration_seconds``.
    """
    num_frames: int | None
    """Number of frames in the stream (int or None).
    This corresponds to ``num_frames_from_content`` if a :term:`scan` was made,
    otherwise it corresponds to ``num_frames_from_header``. If that value is also
    None, the number of frames is calculated from the duration and the average fps.
    """
    average_fps: float | None
    """Average fps of the stream. If a :term:`scan` was perfomed, this is
    computed from the number of frames and the duration of the stream.
    Otherwise we fall back to ``average_fps_from_header``.
    """

    def __repr__(self):
        return super().__repr__()


@dataclass
class AudioStreamHeaderMetadata(StreamMetadata):
    """Metadata of a single audio stream, as reported by the container header."""

    sample_rate: int | None
    """The original sample rate."""
    num_channels: int | None
    """The number of channels (1 for mono, 2 for stereo, etc.)"""
    sample_format: str | None
    """The original sample format, as described by FFmpeg. E.g. 'fltp', 's32', etc."""

    def __repr__(self):
        return super().__repr__()


@dataclass
class AudioStreamMetadata(AudioStreamHeaderMetadata):
    """Metadata of a single audio stream."""

    # Computed fields (computed in C++ with fallback logic)
    duration_seconds: float | None
    """Duration of the stream in seconds. We try to calculate the duration
    from the actual frames if a :term:`scan` was performed. Otherwise we
    fall back to ``duration_seconds_from_header``. If that value is also None,
    we instead calculate the duration from ``num_frames_from_header`` and
    ``average_fps_from_header``. If all of those are unavailable, we fall back
    to the container-level ``duration_seconds_from_header``.
    """
    begin_stream_seconds: float | None
    """Beginning of the stream, in seconds (float). Conceptually, this
    corresponds to the first frame's :term:`pts`. If a :term:`scan` was performed
    and ``begin_stream_seconds_from_content`` is not None, then it is returned.
    Otherwise, this value is 0.
    """

    def __repr__(self):
        return super().__repr__()


@dataclass
class DemuxerMetadata:
    """Container-level metadata, as reported by the header.

    This is what a demuxer can say about the container itself. Everything about
    the streams it follows is on those streams; the full list of streams in the
    file, including the ones that cannot be followed, comes from
    ``get_container_metadata()`` instead.
    """

    duration_seconds_from_header: float | None
    """Duration of the container, in seconds, obtained from the header (float
    or None). Some containers carry a duration only here, with their streams
    reporting none, which is why this is worth having separately."""
    bit_rate_from_header: float | None
    """Overall bit rate of the container (float or None). Not the sum of the
    streams' bit rates: it includes muxing overhead."""
    best_video_stream_index: int | None
    """Index of the stream FFmpeg considers the best video one (int or None)."""
    best_audio_stream_index: int | None
    """Index of the stream FFmpeg considers the best audio one (int or None)."""

    def __repr__(self):
        s = self.__class__.__name__ + ":\n"
        for field in dataclasses.fields(self):
            s += f"{SPACES}{field.name}: {getattr(self, field.name)}\n"
        return s


@dataclass
class ContainerMetadata(DemuxerMetadata):
    streams: list[StreamMetadata]
    """One entry per stream in the file, indexed by stream index. Streams that
    are neither video nor audio (subtitles, data) are plain
    :class:`StreamMetadata`: you can see them, you cannot decode them."""


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
def _stream_metadata_from_dict(stream_dict: dict, stream_index: int) -> StreamMetadata:
    """Build the header-tier metadata for one stream, from its JSON dict.

    Shared by the decoder path, which then adds the content-derived and
    computed fields, and by the building-block path, which has only this tier.
    """
    # Values come out of untyped JSON, so this is Any as far as mypy can tell.
    header_meta: dict[str, Any] = dict(
        duration_seconds_from_header=stream_dict.get("durationSecondsFromHeader"),
        bit_rate=stream_dict.get("bitRate"),
        begin_stream_seconds_from_header=stream_dict.get(
            "beginStreamSecondsFromHeader"
        ),
        codec=stream_dict.get("codec"),
        stream_index=stream_index,
    )
    if stream_dict["mediaType"] == "video":
        return VideoStreamHeaderMetadata(
            width=stream_dict.get("width"),
            height=stream_dict.get("height"),
            num_frames_from_header=stream_dict.get("numFramesFromHeader"),
            average_fps_from_header=stream_dict.get("averageFpsFromHeader"),
            pixel_aspect_ratio=_get_optional_par_fraction(stream_dict),
            rotation=stream_dict.get("rotation"),
            color_primaries=stream_dict.get("colorPrimaries"),
            color_space=stream_dict.get("colorSpace"),
            color_transfer_characteristic=stream_dict.get(
                "colorTransferCharacteristic"
            ),
            pixel_format=stream_dict.get("pixelFormat"),
            **header_meta,
        )
    if stream_dict["mediaType"] == "audio":
        return AudioStreamHeaderMetadata(
            sample_rate=stream_dict.get("sampleRate"),
            num_channels=stream_dict.get("numChannels"),
            sample_format=stream_dict.get("sampleFormat"),
            **header_meta,
        )
    # Neither video nor audio. Could be e.g. subtitles: visible, not decodable.
    return StreamMetadata(**header_meta)


def get_container_metadata(decoder: torch.Tensor) -> ContainerMetadata:
    """Return container metadata from a decoder.

    The accuracy of the metadata and the availability of some returned fields
    depends on whether a full scan was performed by the decoder.
    """

    container_dict = json.loads(_get_container_json_metadata(decoder))
    streams_metadata: list[StreamMetadata] = []
    for stream_index in range(container_dict["numStreams"]):
        stream_dict = json.loads(_get_stream_json_metadata(decoder, stream_index))
        header_meta = dict(
            duration_seconds_from_header=stream_dict.get("durationSecondsFromHeader"),
            bit_rate=stream_dict.get("bitRate"),
            begin_stream_seconds_from_header=stream_dict.get(
                "beginStreamSecondsFromHeader"
            ),
            codec=stream_dict.get("codec"),
            stream_index=stream_index,
        )
        # Only the video and audio classes have the computed fields; a stream of
        # any other type is described by its header alone.
        computed_meta = dict(
            duration_seconds=stream_dict.get("durationSeconds"),
            begin_stream_seconds=stream_dict.get("beginStreamSeconds"),
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
                    end_stream_seconds=stream_dict.get("endStreamSeconds"),
                    num_frames=stream_dict.get("numFrames"),
                    average_fps=stream_dict.get("averageFps"),
                    width=stream_dict.get("width"),
                    height=stream_dict.get("height"),
                    num_frames_from_header=stream_dict.get("numFramesFromHeader"),
                    num_frames_from_content=stream_dict.get("numFramesFromContent"),
                    average_fps_from_header=stream_dict.get("averageFpsFromHeader"),
                    pixel_aspect_ratio=_get_optional_par_fraction(stream_dict),
                    rotation=stream_dict.get("rotation"),
                    color_primaries=stream_dict.get("colorPrimaries"),
                    color_space=stream_dict.get("colorSpace"),
                    color_transfer_characteristic=stream_dict.get(
                        "colorTransferCharacteristic"
                    ),
                    pixel_format=stream_dict.get("pixelFormat"),
                    **header_meta,
                    **computed_meta,
                )
            )
        elif stream_dict["mediaType"] == "audio":
            streams_metadata.append(
                AudioStreamMetadata(
                    sample_rate=stream_dict.get("sampleRate"),
                    num_channels=stream_dict.get("numChannels"),
                    sample_format=stream_dict.get("sampleFormat"),
                    **header_meta,
                    **computed_meta,
                )
            )
        else:
            # This is neither a video nor audio stream. Could be e.g. subtitles.
            # We still need to add a dummy entry so that len(streams_metadata)
            # is consistent with the number of streams.
            streams_metadata.append(StreamMetadata(**header_meta))

    return ContainerMetadata(
        duration_seconds_from_header=container_dict.get("durationSecondsFromHeader"),
        bit_rate_from_header=container_dict.get("bitRate"),
        best_video_stream_index=container_dict.get("bestVideoStreamIndex"),
        best_audio_stream_index=container_dict.get("bestAudioStreamIndex"),
        streams=streams_metadata,
    )


def get_container_metadata_from_header(
    filename: str | pathlib.Path,
) -> ContainerMetadata:
    return get_container_metadata(
        create_from_file(str(filename), seek_mode="approximate")
    )
