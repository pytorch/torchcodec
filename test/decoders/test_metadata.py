import pytest

from torchcodec.decoders._core import (
    create_from_file,
    get_video_metadata,
    scan_all_streams_to_update_metadata,
    StreamMetadata,
)

from ..utils import NASA_VIDEO


def test_get_video_metadata():
    decoder = create_from_file(str(NASA_VIDEO.path))
    scan_all_streams_to_update_metadata(decoder)
    metadata = get_video_metadata(decoder)
    assert len(metadata.streams) == 6
    assert metadata.best_video_stream_index == 3
    assert metadata.best_audio_stream_index == 4

    with pytest.raises(
        NotImplementedError, match="TODO_BEFORE_RELEASE"
    ):
        metadata.duration_seconds
    with pytest.raises(
        NotImplementedError, match="TODO_BEFORE_RELEASE"
    ):
        metadata.bit_rate

    # TODO_BEFORE_RELEASE Nicolas: put these checks back once D58974580 is landed. The expected values
    # are different depending on the FFmpeg version.
    # assert metadata.duration_seconds_container == pytest.approx(16.57, abs=0.001)
    # assert metadata.bit_rate_container == 324915

    best_stream_metadata = metadata.streams[metadata.best_video_stream_index]
    assert best_stream_metadata is metadata.best_video_stream
    assert best_stream_metadata.duration_seconds == pytest.approx(13.013, abs=0.001)
    assert best_stream_metadata.bit_rate == 128783
    assert best_stream_metadata.average_fps == pytest.approx(29.97, abs=0.001)
    assert best_stream_metadata.codec == "h264"
    assert best_stream_metadata.num_frames_computed == 390
    assert best_stream_metadata.num_frames_retrieved == 390


@pytest.mark.parametrize(
    "num_frames_retrieved, num_frames_computed, expected_num_frames",
    [(None, 10, 10), (10, None, 10), (None, None, None)],
)
def test_num_frames_fallback(
    num_frames_retrieved, num_frames_computed, expected_num_frames
):
    """Check that num_frames_computed always has priority when accessing `.num_frames`"""
    metadata = StreamMetadata(
        duration_seconds=4,
        bit_rate=123,
        num_frames_retrieved=num_frames_retrieved,
        num_frames_computed=num_frames_computed,
        min_pts_seconds=0,
        max_pts_seconds=4,
        codec="whatever",
        width=123,
        height=321,
        average_fps=30,
        stream_index=0,
    )

    assert metadata.num_frames == expected_num_frames
