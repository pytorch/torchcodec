import pytest

from torchcodec.decoders._core import create_from_file, get_video_metadata

from ..test_utils import get_reference_video_path


def test_get_video_metadata():
    decoder = create_from_file(str(get_reference_video_path()))
    metadata = get_video_metadata(decoder)
    assert len(metadata.streams) == 6
    assert metadata.best_video_stream_index == 3
    assert metadata.best_audio_stream_index == 3

    assert metadata.duration_seconds_container == pytest.approx(16.57, abs=0.001)
    assert metadata.bit_rate_container == 324915

    best_stream_metadata = metadata.streams[metadata.best_video_stream_index]
    assert best_stream_metadata.duration_seconds == pytest.approx(13.013, abs=0.001)
    assert best_stream_metadata.bit_rate == 128783
    assert best_stream_metadata.average_fps == pytest.approx(29.97, abs=0.001)
    assert best_stream_metadata.codec == "h264"
    assert best_stream_metadata.num_frames_computed == 390
    assert best_stream_metadata.num_frames_retrieved == 390
