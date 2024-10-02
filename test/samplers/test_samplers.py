import contextlib
import random
import re

import pytest
import torch
from torchcodec.decoders import FrameBatch, VideoDecoder
from torchcodec.samplers import clips_at_random_indices

from ..utils import assert_tensor_equal, NASA_VIDEO


@pytest.mark.parametrize("num_indices_between_frames", [1, 5])
def test_random_sampler(num_indices_between_frames):
    decoder = VideoDecoder(NASA_VIDEO.path)
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


@pytest.mark.parametrize(
    "sampling_range_start, sampling_range_end, assert_all_equal",
    (
        (10, 11, True),
        (10, 12, False),
    ),
)
def test_random_sampler_range(
    sampling_range_start, sampling_range_end, assert_all_equal
):
    # Test the sampling_range_start and sampling_range_end parameters by
    # asserting that all clips are equal if the sampling range is of size 1,
    # and that they are not all equal if the sampling range is of size 2.

    # When size=2 there's still a (small) non-zero probability of sampling the
    # same indices for clip starts, so we hard-code a seed that works.
    torch.manual_seed(0)

    decoder = VideoDecoder(NASA_VIDEO.path)

    clips = clips_at_random_indices(
        decoder,
        num_clips=10,
        num_frames_per_clip=2,
        sampling_range_start=sampling_range_start,
        sampling_range_end=sampling_range_end,
    )

    # This context manager is used to ensure that the call to
    # assert_tensor_equal() below either passes (nullcontext) or fails
    # (pytest.raises)
    cm = (
        contextlib.nullcontext()
        if assert_all_equal
        else pytest.raises(AssertionError, match="Tensor-likes are not")
    )
    with cm:
        for clip in clips:
            assert_tensor_equal(clip.data, clips[0].data)


def test_random_sampler_range_negative():
    # Test the passing negative values for sampling_range_start and
    # sampling_range_end is the same as passing `len(decoder) - val`

    decoder = VideoDecoder(NASA_VIDEO.path)

    clips_1 = clips_at_random_indices(
        decoder,
        num_clips=10,
        num_frames_per_clip=2,
        sampling_range_start=len(decoder) - 100,
        sampling_range_end=len(decoder) - 99,
    )

    clips_2 = clips_at_random_indices(
        decoder,
        num_clips=10,
        num_frames_per_clip=2,
        sampling_range_start=-100,
        sampling_range_end=-99,
    )

    # There is only one unique clip in clips_1...
    for clip in clips_1:
        assert_tensor_equal(clip.data, clips_1[0].data)
    # ... and it's the same that's in clips_2
    for clip in clips_2:
        assert_tensor_equal(clip.data, clips_1[0].data)


def test_random_sampler_randomness():
    decoder = VideoDecoder(NASA_VIDEO.path)
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
    with pytest.raises(AssertionError, match="Tensor-likes are not"):
        assert_tensor_equal(clips_1[0].data, clips_3[0].data)

    # Make sure we didn't alter the builtin Python RNG
    builtin_random_state_end = random.getstate()
    assert builtin_random_state_start == builtin_random_state_end


def test_random_sampler_errors():
    decoder = VideoDecoder(NASA_VIDEO.path)
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
        ValueError, match=re.escape("sampling_range_start (1000) must be smaller than")
    ):
        clips_at_random_indices(decoder, sampling_range_start=1000)

    with pytest.raises(
        ValueError, match=re.escape("sampling_range_start (4) must be smaller than")
    ):
        clips_at_random_indices(decoder, sampling_range_start=4, sampling_range_end=4)

    with pytest.raises(
        ValueError, match=re.escape("sampling_range_start (290) must be smaller than")
    ):
        clips_at_random_indices(
            decoder, sampling_range_start=-100, sampling_range_end=-100
        )

    with pytest.raises(
        ValueError, match="We determined that sampling_range_end should"
    ):
        clips_at_random_indices(
            decoder,
            num_frames_per_clip=10,
            sampling_range_start=len(decoder) - 1,
            sampling_range_end=None,
        )
