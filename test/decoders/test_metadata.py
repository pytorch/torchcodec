import pytest

from torchcodec.decoders._core import (
    create_from_file,
    get_ffmpeg_library_versions,
    get_video_metadata,
    scan_all_streams_to_update_metadata,
    VideoStreamMetadata,
)

from ..utils import NASA_VIDEO


def test_get_video_metadata():
    decoder = create_from_file(str(NASA_VIDEO.path))
    scan_all_streams_to_update_metadata(decoder)
    metadata = get_video_metadata(decoder)
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

    best_stream_metadata = metadata.streams[metadata.best_video_stream_index]
    assert best_stream_metadata is metadata.best_video_stream
    assert best_stream_metadata.duration_seconds == pytest.approx(13.013, abs=0.001)
    assert best_stream_metadata.bit_rate == 128783
    assert best_stream_metadata.average_fps == pytest.approx(29.97, abs=0.001)
    assert best_stream_metadata.codec == "h264"
    assert best_stream_metadata.num_frames_from_content == 390
    assert best_stream_metadata.num_frames_from_header == 390


@pytest.mark.parametrize(
    "num_frames_from_header, num_frames_from_content, expected_num_frames",
    [(None, 10, 10), (10, None, 10), (None, None, None)],
)
def test_num_frames_fallback(
    num_frames_from_header, num_frames_from_content, expected_num_frames
):
    """Check that num_frames_from_content always has priority when accessing `.num_frames`"""
    metadata = VideoStreamMetadata(
        duration_seconds=4,
        bit_rate=123,
        num_frames_from_header=num_frames_from_header,
        num_frames_from_content=num_frames_from_content,
        min_pts_seconds=0,
        max_pts_seconds=4,
        codec="whatever",
        width=123,
        height=321,
        average_fps=30,
        stream_index=0,
    )

    assert metadata.num_frames == expected_num_frames
