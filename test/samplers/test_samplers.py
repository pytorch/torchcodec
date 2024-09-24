import random
import re

import pytest
import torch
from torchcodec.decoders import FrameBatch, SimpleVideoDecoder
from torchcodec.samplers import clips_at_random_indices

from ..utils import assert_tensor_equal, NASA_VIDEO


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


def test_random_sampler_randomness():
    decoder = SimpleVideoDecoder(NASA_VIDEO.path)
    num_clips = 5

    builtin_random_state_start = random.getstate()

    torch.manual_seed(0)
    clips_1 = clips_at_random_indices(decoder, num_clips=num_clips)

    # Assert the clip starts aren't sorted, to make sure we haven't messed up
    # the implementation. (This may fail if we're unlucky, but we hard-coded a
    # seed, so it will always pass.)
    clip_starts = [clip.pts_seconds.item() for clip in clips_1]
    assert sorted(clip_starts) != clip_starts

    # Call the same sampler again with the same seed, expect same results
    torch.manual_seed(0)
    clips_2 = clips_at_random_indices(decoder, num_clips=num_clips)
    for clip_1, clip_2 in zip(clips_1, clips_2):
        assert_tensor_equal(clip_1.data, clip_2.data)
        assert_tensor_equal(clip_1.pts_seconds, clip_2.pts_seconds)
        assert_tensor_equal(clip_1.duration_seconds, clip_2.duration_seconds)

    # Call with a different seed, expect different results
    torch.manual_seed(1)
    clips_3 = clips_at_random_indices(decoder, num_clips=num_clips)
    with pytest.raises(AssertionError, match="not equal"):
        assert_tensor_equal(clips_1[0].data, clips_3[0].data)

    # Make sure we didn't alter the builting Python RNG
    builtin_random_state_end = random.getstate()
    assert builtin_random_state_start == builtin_random_state_end


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

    with pytest.raises(
        ValueError, match=re.escape("sampling_range_start (-1) must be non-negative")
    ):
        clips_at_random_indices(decoder, sampling_range_start=-1)

    with pytest.raises(
        ValueError, match=re.escape("sampling_range_start (4) must be smaller than")
    ):
        clips_at_random_indices(decoder, sampling_range_start=4, sampling_range_end=0)

    with pytest.raises(
        ValueError, match="We determined that sampling_range_end should"
    ):
        clips_at_random_indices(
            decoder,
            num_frames_per_clip=10,
            sampling_range_start=len(decoder) - 1,
            sampling_range_end=None,
        )
