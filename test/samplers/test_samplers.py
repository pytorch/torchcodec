import re

import pytest
import torch
from torchcodec.decoders import FrameBatch, SimpleVideoDecoder
from torchcodec.samplers import clips_at_random_indices

from ..utils import NASA_VIDEO


@pytest.mark.parametrize("num_indices_between_frames", [1, 5])
def test_random_sampler(num_indices_between_frames):
    decoder = SimpleVideoDecoder(NASA_VIDEO.path)
    num_clips = 2
    num_frames_per_clip = 3

    clips = clips_at_random_indices(
        decoder,
        num_clips=num_clips,
        num_frames_per_clip=num_frames_per_clip,
        num_indices_between_frames=num_indices_between_frames,
    )

    assert isinstance(clips, list)
    assert len(clips) == num_clips
    assert all(isinstance(clip, FrameBatch) for clip in clips)
    expected_clip_data_shape = (
        num_frames_per_clip,
        3,
        NASA_VIDEO.height,
        NASA_VIDEO.width,
    )
    assert all(clip.data.shape == expected_clip_data_shape for clip in clips)

    # Check the num_indices_between_frames parameter by asserting that the
    # "time" difference between frames in a clip is the same as the "index"
    # distance.
    avg_distance_between_frames_seconds = torch.concat(
        [clip.pts_seconds.diff() for clip in clips]
    ).mean()
    assert avg_distance_between_frames_seconds == pytest.approx(
        num_indices_between_frames / decoder.metadata.average_fps
    )


def test_random_sampler_errors():
    decoder = SimpleVideoDecoder(NASA_VIDEO.path)
    with pytest.raises(
        ValueError, match=re.escape("num_clips (0) must be strictly positive")
    ):
        clips_at_random_indices(decoder, num_clips=0)

    with pytest.raises(
        ValueError, match=re.escape("num_frames_per_clip (0) must be strictly positive")
    ):
        clips_at_random_indices(decoder, num_frames_per_clip=0)

    with pytest.raises(
        ValueError,
        match=re.escape("num_indices_between_frames (0) must be strictly positive"),
    ):
        clips_at_random_indices(decoder, num_indices_between_frames=0)

    with pytest.raises(
        ValueError,
        match=re.escape("Clip span (1000) is larger than the number of frames"),
    ):
        clips_at_random_indices(decoder, num_frames_per_clip=1000)

    with pytest.raises(
        ValueError,
        match=re.escape("Clip span (1001) is larger than the number of frames"),
    ):
        clips_at_random_indices(
            decoder, num_frames_per_clip=2, num_indices_between_frames=1000
        )
