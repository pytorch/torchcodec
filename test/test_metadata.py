# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
from fractions import Fraction

import pytest
from torchcodec import ffmpeg_major_version
from torchcodec._core import (
    AudioStreamMetadata,
    get_container_metadata,
    get_container_metadata_from_header,
    VideoStreamMetadata,
)
from torchcodec._core.ops import add_video_stream, create_from_file
from torchcodec.decoders import AudioDecoder, VideoDecoder

from .utils import (
    BT2020_LIMITED_RANGE_10BIT,
    NASA_AUDIO_MP3,
    NASA_VIDEO,
    NASA_VIDEO_ROTATED,
)


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
        pytest.param(
            functools.partial(
                _get_container_metadata, seek_mode="custom_frame_mappings"
            ),
            marks=pytest.mark.skipif(
                ffmpeg_major_version in (4, 5),
                reason="ffprobe isn't accurate on ffmpeg 4 and 5",
            ),
        ),
    ),
)
def test_get_metadata(metadata_getter):
    seek_mode = (
        metadata_getter.keywords["seek_mode"]
        if isinstance(metadata_getter, functools.partial)
        else None
    )
    metadata = metadata_getter(NASA_VIDEO.path)

    with_scan = (
        (seek_mode == "exact" or seek_mode == "custom_frame_mappings")
        if isinstance(metadata_getter, functools.partial)
        else False
    )

    assert len(metadata.streams) == 6
    assert metadata.best_video_stream_index == 3
    assert metadata.best_audio_stream_index == 4

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
    assert best_video_stream_metadata.duration_seconds == pytest.approx(
        13.013, abs=0.001
    )
    assert best_video_stream_metadata.begin_stream_seconds_from_header == 0
    assert best_video_stream_metadata.bit_rate == 128783
    assert best_video_stream_metadata.average_fps == pytest.approx(29.97, abs=0.001)
    assert best_video_stream_metadata.pixel_aspect_ratio == Fraction(1, 1)
    assert best_video_stream_metadata.pixel_format == "yuv420p"
    assert best_video_stream_metadata.codec == "h264"
    assert best_video_stream_metadata.num_frames_from_content == (
        390 if with_scan else None
    )
    assert best_video_stream_metadata.num_frames_from_header == 390
    assert best_video_stream_metadata.num_frames == 390

    best_audio_stream_metadata = metadata.streams[metadata.best_audio_stream_index]
    assert isinstance(best_audio_stream_metadata, AudioStreamMetadata)
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

    expected_duration_seconds_from_header = (
        13.056 if ffmpeg_major_version >= 8 else 13.248
    )

    assert (
        best_audio_stream_metadata.duration_seconds_from_header
        == expected_duration_seconds_from_header
    )
    assert best_audio_stream_metadata.begin_stream_seconds_from_header == 0.138125
    assert best_audio_stream_metadata.bit_rate == 64000
    assert best_audio_stream_metadata.codec == "mp3"
    assert best_audio_stream_metadata.sample_format == "fltp"


def test_rotation_metadata():
    """Test that rotation metadata is correctly extracted for rotated video."""
    # NASA_VIDEO_ROTATED has 90-degree rotation metadata
    decoder_rotated = VideoDecoder(NASA_VIDEO_ROTATED.path)
    assert decoder_rotated.metadata.rotation is not None
    assert decoder_rotated.metadata.rotation == 90

    # NASA_VIDEO has no rotation metadata
    decoder = VideoDecoder(NASA_VIDEO.path)
    assert decoder.metadata.rotation is None

    # Check that height and width are reported post-rotation
    # For 90-degree rotation, width and height should be swapped
    assert (decoder_rotated.metadata.height, decoder_rotated.metadata.width) == (
        decoder.metadata.width,
        decoder.metadata.height,
    )


def test_color_metadata():
    # BT2020_LIMITED_RANGE_10BIT is a BT.2020 10-bit HEVC video with PQ transfer
    decoder_bt2020 = VideoDecoder(BT2020_LIMITED_RANGE_10BIT.path)
    assert decoder_bt2020.metadata.color_primaries == "bt2020"
    assert decoder_bt2020.metadata.color_space == "bt2020nc"
    assert decoder_bt2020.metadata.color_transfer_characteristic == "smpte2084"
    assert decoder_bt2020.metadata.pixel_format == "yuv420p10le"

    # NASA_VIDEO has BT.709 color metadata
    decoder_nasa = VideoDecoder(NASA_VIDEO.path)
    assert decoder_nasa.metadata.color_primaries == "bt709"
    assert decoder_nasa.metadata.color_space == "bt709"
    assert decoder_nasa.metadata.color_transfer_characteristic == "bt709"
    assert decoder_nasa.metadata.pixel_format == "yuv420p"


def test_repr():
    # Test for calls to print(), str(), etc. Useful to make sure we don't forget
    # to add additional @properties to __repr__
    assert (
        str(VideoDecoder(NASA_VIDEO.path).metadata)
        == """VideoStreamMetadata:
  duration_seconds_from_header: 13.013
  begin_stream_seconds_from_header: 0
  bit_rate: 128783
  codec: h264
  stream_index: 3
  width: 480
  height: 270
  num_frames_from_header: 390
  average_fps_from_header: 29.97002997002997
  pixel_aspect_ratio: 1
  rotation: None
  color_primaries: bt709
  color_space: bt709
  color_transfer_characteristic: bt709
  pixel_format: yuv420p
  begin_stream_seconds_from_content: 0
  end_stream_seconds_from_content: 13.013
  num_frames_from_content: 390
  duration_seconds: 13.013
  begin_stream_seconds: 0
  end_stream_seconds: 13.013
  num_frames: 390
  average_fps: 29.97002997002997
"""
    )
    expected_duration_seconds_from_header = (
        13.056 if ffmpeg_major_version >= 8 else 13.248
    )

    assert (
        str(AudioDecoder(NASA_AUDIO_MP3.path).metadata)
        == f"""AudioStreamMetadata:
  duration_seconds_from_header: {expected_duration_seconds_from_header}
  begin_stream_seconds_from_header: 0.138125
  bit_rate: 64000
  codec: mp3
  stream_index: 0
  sample_rate: 8000
  num_channels: 2
  sample_format: fltp
  duration_seconds: {expected_duration_seconds_from_header}
  begin_stream_seconds: 0.138125
"""
    )
