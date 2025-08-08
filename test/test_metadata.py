# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
from fractions import Fraction

import pytest

from torchcodec._core import (
    add_video_stream,
    AudioStreamMetadata,
    create_from_file,
    get_container_metadata,
    get_container_metadata_from_header,
    get_ffmpeg_library_versions,
    VideoStreamMetadata,
)
from torchcodec.decoders import AudioDecoder, VideoDecoder

from .utils import get_ffmpeg_major_version, NASA_AUDIO_MP3, NASA_VIDEO


# TODO: Expected values in these tests should be based on the assets's
# attributes rather than on hard-coded values.


def _get_container_metadata(path, seek_mode):
    decoder = create_from_file(str(path), seek_mode=seek_mode)

    # For custom_frame_mappings seek mode, add a video stream to update metadata
    if seek_mode == "custom_frame_mappings":
        custom_frame_mappings = NASA_VIDEO.get_custom_frame_mappings()

        # Add the best video stream (index 3 for NASA_VIDEO)
        add_video_stream(
            decoder,
            stream_index=NASA_VIDEO.default_stream_index,
            custom_frame_mappings=custom_frame_mappings,
        )
    return get_container_metadata(decoder)


@pytest.mark.parametrize(
    "metadata_getter",
    (
        get_container_metadata_from_header,
        functools.partial(_get_container_metadata, seek_mode="approximate"),
        functools.partial(_get_container_metadata, seek_mode="exact"),
        functools.partial(_get_container_metadata, seek_mode="custom_frame_mappings"),
    ),
)
def test_get_metadata(metadata_getter):
    seek_mode = (
        metadata_getter.keywords["seek_mode"]
        if isinstance(metadata_getter, functools.partial)
        else None
    )
    if (seek_mode == "custom_frame_mappings") and get_ffmpeg_major_version() in (4, 5):
        pytest.skip(reason="ffprobe isn't accurate on ffmpeg 4 and 5")
    with_added_video_stream = seek_mode == "custom_frame_mappings"
    metadata = metadata_getter(NASA_VIDEO.path)

    with_scan = (
        (seek_mode == "exact" or seek_mode == "custom_frame_mappings")
        if isinstance(metadata_getter, functools.partial)
        else False
    )

    assert len(metadata.streams) == 6
    assert metadata.best_video_stream_index == 3
    assert metadata.best_audio_stream_index == 4

    with pytest.raises(NotImplementedError, match="Decide on logic"):
        metadata.duration_seconds
    with pytest.raises(NotImplementedError, match="Decide on logic"):
        metadata.bit_rate

    ffmpeg_major_version = get_ffmpeg_major_version()
    if ffmpeg_major_version <= 5:
        expected_duration_seconds_from_header = 16.57
        expected_bit_rate_from_header = 324915
    else:
        expected_duration_seconds_from_header = 13.056
        expected_bit_rate_from_header = 412365

    assert (
        metadata.duration_seconds_from_header == expected_duration_seconds_from_header
    )
    assert metadata.bit_rate_from_header == expected_bit_rate_from_header

    best_video_stream_metadata = metadata.streams[metadata.best_video_stream_index]
    assert isinstance(best_video_stream_metadata, VideoStreamMetadata)
    assert best_video_stream_metadata is metadata.best_video_stream
    assert best_video_stream_metadata.duration_seconds == pytest.approx(
        13.013, abs=0.001
    )
    assert best_video_stream_metadata.begin_stream_seconds_from_header == 0
    assert best_video_stream_metadata.bit_rate == 128783
    assert best_video_stream_metadata.average_fps == pytest.approx(29.97, abs=0.001)
    assert best_video_stream_metadata.pixel_aspect_ratio == (
        Fraction(1, 1) if with_added_video_stream else None
    )
    assert best_video_stream_metadata.codec == "h264"
    assert best_video_stream_metadata.num_frames_from_content == (
        390 if with_scan else None
    )
    assert best_video_stream_metadata.num_frames_from_header == 390
    assert best_video_stream_metadata.num_frames == 390

    best_audio_stream_metadata = metadata.streams[metadata.best_audio_stream_index]
    assert isinstance(best_audio_stream_metadata, AudioStreamMetadata)
    assert best_audio_stream_metadata is metadata.best_audio_stream
    assert best_audio_stream_metadata.duration_seconds_from_header == 13.056
    assert best_audio_stream_metadata.begin_stream_seconds_from_header == 0
    assert best_audio_stream_metadata.bit_rate == 128837
    assert best_audio_stream_metadata.codec == "aac"
    assert best_audio_stream_metadata.sample_format == "fltp"


@pytest.mark.parametrize(
    "metadata_getter",
    (
        get_container_metadata_from_header,
        functools.partial(_get_container_metadata, seek_mode="approximate"),
    ),
)
def test_get_metadata_audio_file(metadata_getter):
    metadata = metadata_getter(NASA_AUDIO_MP3.path)
    best_audio_stream_metadata = metadata.streams[metadata.best_audio_stream_index]
    assert isinstance(best_audio_stream_metadata, AudioStreamMetadata)
    assert best_audio_stream_metadata is metadata.best_audio_stream
    assert best_audio_stream_metadata.duration_seconds_from_header == 13.248
    assert best_audio_stream_metadata.begin_stream_seconds_from_header == 0.138125
    assert best_audio_stream_metadata.bit_rate == 64000
    assert best_audio_stream_metadata.codec == "mp3"
    assert best_audio_stream_metadata.sample_format == "fltp"


@pytest.mark.parametrize(
    "num_frames_from_header, num_frames_from_content, expected_num_frames",
    [(10, 20, 20), (None, 10, 10), (10, None, 10)],
)
def test_num_frames_fallback(
    num_frames_from_header, num_frames_from_content, expected_num_frames
):
    """Check that num_frames_from_content always has priority when accessing `.num_frames`"""
    metadata = VideoStreamMetadata(
        duration_seconds_from_header=4,
        bit_rate=123,
        num_frames_from_header=num_frames_from_header,
        num_frames_from_content=num_frames_from_content,
        begin_stream_seconds_from_header=0,
        begin_stream_seconds_from_content=0,
        end_stream_seconds_from_content=4,
        codec="whatever",
        width=123,
        height=321,
        average_fps_from_header=30,
        pixel_aspect_ratio=Fraction(1, 1),
        stream_index=0,
    )

    assert metadata.num_frames == expected_num_frames


@pytest.mark.parametrize(
    "average_fps_from_header, duration_seconds_from_header, expected_num_frames",
    [(60, 10, 600), (60, None, None), (None, 10, None), (None, None, None)],
)
def test_calculate_num_frames_using_fps_and_duration(
    average_fps_from_header, duration_seconds_from_header, expected_num_frames
):
    """Check that if num_frames_from_content and num_frames_from_header are missing,
    `.num_frames` is calculated using average_fps_from_header and duration_seconds_from_header
    """
    metadata = VideoStreamMetadata(
        duration_seconds_from_header=duration_seconds_from_header,
        bit_rate=123,
        num_frames_from_header=None,  # None to test calculating num_frames
        num_frames_from_content=None,  # None to test calculating num_frames
        begin_stream_seconds_from_header=0,
        begin_stream_seconds_from_content=0,
        end_stream_seconds_from_content=4,
        codec="whatever",
        width=123,
        height=321,
        pixel_aspect_ratio=Fraction(10, 11),
        average_fps_from_header=average_fps_from_header,
        stream_index=0,
    )

    assert metadata.num_frames == expected_num_frames


@pytest.mark.parametrize(
    "duration_seconds_from_header, begin_stream_seconds_from_content, end_stream_seconds_from_content, expected_duration_seconds",
    [(60, 5, 20, 15), (60, 1, None, 60), (60, None, 1, 60), (None, 0, 10, 10)],
)
def test_duration_seconds_fallback(
    duration_seconds_from_header,
    begin_stream_seconds_from_content,
    end_stream_seconds_from_content,
    expected_duration_seconds,
):
    """Check that using begin_stream_seconds_from_content and end_stream_seconds_from_content to calculate `.duration_seconds`
    has priority. If either value is missing, duration_seconds_from_header is used.
    """
    metadata = VideoStreamMetadata(
        duration_seconds_from_header=duration_seconds_from_header,
        bit_rate=123,
        num_frames_from_header=5,
        num_frames_from_content=10,
        begin_stream_seconds_from_header=0,
        begin_stream_seconds_from_content=begin_stream_seconds_from_content,
        end_stream_seconds_from_content=end_stream_seconds_from_content,
        codec="whatever",
        width=123,
        height=321,
        pixel_aspect_ratio=Fraction(10, 11),
        average_fps_from_header=5,
        stream_index=0,
    )

    assert metadata.duration_seconds == expected_duration_seconds


@pytest.mark.parametrize(
    "num_frames_from_header, average_fps_from_header, expected_duration_seconds",
    [(100, 10, 10), (100, None, None), (None, 10, None), (None, None, None)],
)
def test_calculate_duration_seconds_using_fps_and_num_frames(
    num_frames_from_header, average_fps_from_header, expected_duration_seconds
):
    """Check that duration_seconds is calculated using average_fps_from_header and num_frames_from_header
    if duration_seconds_from_header is missing.
    """
    metadata = VideoStreamMetadata(
        duration_seconds_from_header=None,  # None to test calculating duration_seconds
        bit_rate=123,
        num_frames_from_header=num_frames_from_header,
        num_frames_from_content=10,
        begin_stream_seconds_from_header=0,
        begin_stream_seconds_from_content=None,  # None to test calculating duration_seconds
        end_stream_seconds_from_content=None,  # None to test calculating duration_seconds
        codec="whatever",
        width=123,
        height=321,
        pixel_aspect_ratio=Fraction(10, 11),
        average_fps_from_header=average_fps_from_header,
        stream_index=0,
    )
    assert metadata.duration_seconds_from_header is None
    assert metadata.duration_seconds == expected_duration_seconds


def test_repr():
    # Test for calls to print(), str(), etc. Useful to make sure we don't forget
    # to add additional @properties to __repr__
    assert (
        str(VideoDecoder(NASA_VIDEO.path).metadata)
        == """VideoStreamMetadata:
  duration_seconds_from_header: 13.013
  begin_stream_seconds_from_header: 0.0
  bit_rate: 128783.0
  codec: h264
  stream_index: 3
  begin_stream_seconds_from_content: 0.0
  end_stream_seconds_from_content: 13.013
  width: 480
  height: 270
  num_frames_from_header: 390
  num_frames_from_content: 390
  average_fps_from_header: 29.97003
  pixel_aspect_ratio: 1
  duration_seconds: 13.013
  begin_stream_seconds: 0.0
  end_stream_seconds: 13.013
  num_frames: 390
  average_fps: 29.97002997002997
"""
    )

    assert (
        str(AudioDecoder(NASA_AUDIO_MP3.path).metadata)
        == """AudioStreamMetadata:
  duration_seconds_from_header: 13.248
  begin_stream_seconds_from_header: 0.138125
  bit_rate: 64000.0
  codec: mp3
  stream_index: 0
  sample_rate: 8000
  num_channels: 2
  sample_format: fltp
"""
    )
