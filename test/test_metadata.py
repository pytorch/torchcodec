# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools

import pytest

from torchcodec._core import (
    AudioStreamMetadata,
    create_from_file,
    get_container_metadata,
    get_container_metadata_from_header,
    get_ffmpeg_library_versions,
    VideoStreamMetadata,
)
from torchcodec.decoders import AudioDecoder, VideoDecoder

from .utils import NASA_AUDIO_MP3, NASA_VIDEO


# TODO: Expected values in these tests should be based on the assets's
# attributes rather than on hard-coded values.


def _get_container_metadata(path, seek_mode):
    decoder = create_from_file(str(path), seek_mode=seek_mode)
    return get_container_metadata(decoder)


@pytest.mark.parametrize(
    "metadata_getter",
    (
        get_container_metadata_from_header,
        functools.partial(_get_container_metadata, seek_mode="approximate"),
        functools.partial(_get_container_metadata, seek_mode="exact"),
    ),
)
def test_get_metadata(metadata_getter):
    with_scan = (
        metadata_getter.keywords["seek_mode"] == "exact"
        if isinstance(metadata_getter, functools.partial)
        else False
    )

    metadata = metadata_getter(NASA_VIDEO.path)
    # metadata = metadata_getter(NASA_VIDEO.path)

    assert len(metadata.streams) == 6
    assert metadata.best_video_stream_index == 3
    assert metadata.best_audio_stream_index == 4

    with pytest.raises(NotImplementedError, match="Decide on logic"):
        metadata.duration_seconds
    with pytest.raises(NotImplementedError, match="Decide on logic"):
        metadata.bit_rate

    ffmpeg_major_version = int(
        get_ffmpeg_library_versions()["ffmpeg_version"].split(".")[0]
    )
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
        average_fps_from_header=average_fps_from_header,
        stream_index=0,
    )

    assert metadata.num_frames == expected_num_frames


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
