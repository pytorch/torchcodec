import contextlib
import random
import re
from copy import copy
from functools import partial

import pytest
import torch

from torchcodec import FrameBatch
from torchcodec.decoders import VideoDecoder
from torchcodec.samplers import (
    clips_at_random_indices,
    clips_at_random_timestamps,
    clips_at_regular_indices,
    clips_at_regular_timestamps,
)
from torchcodec.samplers._common import _POLICY_FUNCTIONS
from torchcodec.samplers._index_based import _build_all_clips_indices
from torchcodec.samplers._time_based import _build_all_clips_timestamps

from .utils import assert_frames_equal, H265_10BITS, NASA_VIDEO


def _assert_output_type_and_shapes(
    video, clips, expected_num_clips, num_frames_per_clip
):
    assert isinstance(clips, FrameBatch)
    assert clips.data.shape == (
        expected_num_clips,
        num_frames_per_clip,
        3,
        video.height,
        video.width,
    )
    assert clips.pts_seconds.shape == (
        expected_num_clips,
        num_frames_per_clip,
    )
    assert clips.duration_seconds.shape == (
        expected_num_clips,
        num_frames_per_clip,
    )


def _assert_regular_sampler(clips, expected_seconds_between_clip_starts=None):
    seconds_between_clip_starts = clips.pts_seconds[:, 0].diff()

    if expected_seconds_between_clip_starts is not None:
        # This can only be asserted with the time-based sampler, where
        # seconds_between_clip_starts is known since it's passed by the user.
        torch.testing.assert_close(
            seconds_between_clip_starts,
            expected_seconds_between_clip_starts,
            atol=0.05,
            rtol=0,
        )

    # Also assert clips are sorted by start time
    assert (seconds_between_clip_starts > 0).all()
    # For all samplers, we can at least assert that seconds_between_clip_starts
    # is constant.
    for diff in seconds_between_clip_starts:
        assert diff == pytest.approx(seconds_between_clip_starts[0], abs=0.05)


@pytest.mark.parametrize("sampler", (clips_at_random_indices, clips_at_regular_indices))
@pytest.mark.parametrize("num_indices_between_frames", [1, 5])
def test_index_based_sampler(sampler, num_indices_between_frames):
    decoder = VideoDecoder(NASA_VIDEO.path)
    num_clips = 5
    num_frames_per_clip = 3

    clips = sampler(
        decoder,
        num_clips=num_clips,
        num_frames_per_clip=num_frames_per_clip,
        num_indices_between_frames=num_indices_between_frames,
    )

    _assert_output_type_and_shapes(
        video=NASA_VIDEO,
        clips=clips,
        expected_num_clips=num_clips,
        num_frames_per_clip=num_frames_per_clip,
    )

    if sampler is clips_at_regular_indices:
        _assert_regular_sampler(clips=clips, expected_seconds_between_clip_starts=None)

    # Check the num_indices_between_frames parameter by asserting that the
    # "time" difference between frames in a clip is the same as the "index"
    # distance.
    for clip in clips:
        avg_distance_between_frames_seconds = clip.pts_seconds.diff().mean()
        assert avg_distance_between_frames_seconds == pytest.approx(
            num_indices_between_frames / decoder.metadata.average_fps, abs=1e-5
        )


@pytest.mark.parametrize(
    "sampler",
    (
        partial(clips_at_random_timestamps, num_clips=5),
        partial(clips_at_regular_timestamps, seconds_between_clip_starts=2),
    ),
)
@pytest.mark.parametrize("seconds_between_frames", [None, 3])
def test_time_based_sampler(sampler, seconds_between_frames):
    decoder = VideoDecoder(NASA_VIDEO.path)
    num_frames_per_clip = 3

    clips = sampler(
        decoder,
        num_frames_per_clip=num_frames_per_clip,
        seconds_between_frames=seconds_between_frames,
    )

    expected_num_clips = (
        len(clips)  # No-op check, we can't assert with regular sampler
        if sampler.func is clips_at_regular_timestamps
        else sampler.keywords["num_clips"]
    )
    _assert_output_type_and_shapes(
        video=NASA_VIDEO,
        clips=clips,
        expected_num_clips=expected_num_clips,
        num_frames_per_clip=num_frames_per_clip,
    )

    if sampler.func is clips_at_regular_timestamps:
        seconds_between_clip_starts = sampler.keywords["seconds_between_clip_starts"]
        expected_seconds_between_clip_starts = torch.tensor(
            [seconds_between_clip_starts] * (len(clips) - 1), dtype=torch.float64
        )
        _assert_regular_sampler(
            clips=clips,
            expected_seconds_between_clip_starts=expected_seconds_between_clip_starts,
        )

    expected_seconds_between_frames = (
        seconds_between_frames or 1 / decoder.metadata.average_fps
    )
    for clip in clips:
        avg_seconds_between_frames = clip.pts_seconds.diff().mean()
        assert avg_seconds_between_frames == pytest.approx(
            expected_seconds_between_frames, abs=0.05
        )


@pytest.mark.parametrize(
    "sampler",
    (
        partial(
            clips_at_regular_indices,
            num_clips=1,
            sampling_range_start=0,
            sampling_range_end=1,
        ),
        partial(
            clips_at_random_indices,
            num_clips=1,
            sampling_range_start=0,
            sampling_range_end=1,
        ),
        partial(
            clips_at_random_timestamps,
            num_clips=1,
            sampling_range_start=0,
            sampling_range_end=0.01,
        ),
        partial(
            clips_at_regular_timestamps,
            seconds_between_clip_starts=1,
            seconds_between_frames=0.0335,  # forces consecutive frames
            sampling_range_start=0,
            sampling_range_end=0.01,
        ),
    ),
)
def test_against_ref(sampler):
    # Force the sampler to sample a clip containing the first 5 frames of the
    # video. We can then assert the exact frame values against our existing test
    # resource reference.
    decoder = VideoDecoder(NASA_VIDEO.path)

    num_frames_per_clip = 5
    expected_clip_data = NASA_VIDEO.get_frame_data_by_range(
        start=0, stop=num_frames_per_clip
    )

    clip = sampler(decoder, num_frames_per_clip=num_frames_per_clip)[0]
    assert_frames_equal(clip.data, expected_clip_data)


@pytest.mark.parametrize(
    "sampler, sampling_range_start, sampling_range_end, assert_all_equal",
    (
        (partial(clips_at_random_indices, num_clips=10), 10, 11, True),
        (partial(clips_at_random_indices, num_clips=10), 10, 12, False),
        (partial(clips_at_regular_indices, num_clips=10), 10, 11, True),
        (partial(clips_at_regular_indices, num_clips=10), 10, 12, False),
        (
            partial(clips_at_regular_timestamps, seconds_between_clip_starts=1),
            10.0,
            11.0,
            True,
        ),
        (
            partial(clips_at_regular_timestamps, seconds_between_clip_starts=1),
            10.0,
            12.0,
            False,
        ),
        (partial(clips_at_random_indices, num_clips=10), 10, 11, True),
        (partial(clips_at_random_indices, num_clips=10), 10, 12, False),
    ),
)
def test_sampling_range(
    sampler, sampling_range_start, sampling_range_end, assert_all_equal
):
    # For index-based:
    # Test the sampling_range_start and sampling_range_end parameters by
    # asserting that all clips are equal if the sampling range is of size 1,
    # and that they are not all equal if the sampling range is of size 2.
    #
    # For time-based:
    # The test is similar but with different semantics. We set the sampling
    # range to be 1 second or 2 seconds. Since we set
    # seconds_between_clip_starts to 1 we expect exactly one clip with the
    # sampling range is of size 1, and 2 different clips when the sampling range
    # is 2 seconds.

    # When size=2 there's still a (small) non-zero probability of sampling the
    # same indices for clip starts, so we hard-code a seed that works.
    torch.manual_seed(0)

    decoder = VideoDecoder(NASA_VIDEO.path)

    clips = sampler(
        decoder,
        num_frames_per_clip=2,
        sampling_range_start=sampling_range_start,
        sampling_range_end=sampling_range_end,
    )

    # This context manager is used to ensure that the call to
    # assert_frames_equal() below either passes (nullcontext) or fails
    # (pytest.raises)
    cm = (
        contextlib.nullcontext()
        if assert_all_equal
        else pytest.raises(AssertionError, match="low psnr")
    )
    with cm:
        for clip in clips:
            assert_frames_equal(clip.data, clips[0].data, psnr=float("inf"))


@pytest.mark.parametrize("sampler", (clips_at_random_indices, clips_at_regular_indices))
def test_sampling_range_negative(sampler):
    # Test the passing negative values for sampling_range_start and
    # sampling_range_end is the same as passing `len(decoder) - val`

    decoder = VideoDecoder(NASA_VIDEO.path)

    clips_1 = sampler(
        decoder,
        num_clips=10,
        num_frames_per_clip=2,
        sampling_range_start=len(decoder) - 100,
        sampling_range_end=len(decoder) - 99,
    )

    clips_2 = sampler(
        decoder,
        num_clips=10,
        num_frames_per_clip=2,
        sampling_range_start=-100,
        sampling_range_end=-99,
    )

    # There is only one unique clip in clips_1...
    for clip in clips_1:
        assert_frames_equal(clip.data, clips_1[0].data)
    # ... and it's the same that's in clips_2
    for clip in clips_2:
        assert_frames_equal(clip.data, clips_1[0].data)


@pytest.mark.parametrize(
    "sampler",
    (
        clips_at_random_indices,
        clips_at_random_timestamps,
    ),
)
def test_sampling_range_default_behavior_random_sampler(sampler):
    # This is a functional test for the default behavior of the
    # sampling_range_end parameter, for the random sampler.
    # By default it's None, which means the
    # sampler automatically sets its value such that we never sample "beyond"
    # the number of frames in the video. That means that the last few frames of
    # the video are less likely to be part of a clip.
    # When sampling_range_end is set manually to e.g. len(decoder), the last
    # frames are way more likely to be part of a clip, since there is no
    # restriction on the sampling range (and the user-defined policy comes into
    # action, potentially repeating that last frame).
    #
    # In this test we assert that the last clip starts significantly earlier
    # when sampling_range_end=None than when sampling_range_end=len(decoder).
    # This is only a proxy, for lack of better testing opportunities.

    torch.manual_seed(0)

    decoder = VideoDecoder(NASA_VIDEO.path)

    num_clips = 20
    num_frames_per_clip = 15
    sampling_range_start = -20 if sampler is clips_at_random_indices else 11

    # with default sampling_range_end value
    clips_default = sampler(
        decoder,
        num_clips=num_clips,
        num_frames_per_clip=num_frames_per_clip,
        sampling_range_start=sampling_range_start,
        sampling_range_end=None,
        policy="error",
    )

    # last_clip_start_default = max([clip.pts_seconds[0] for clip in clips_default])
    last_clip_start_default = clips_default.pts_seconds[:, 0].max()

    # with manual sampling_range_end value set to last frame / end of video
    clips_manual = sampler(
        decoder,
        num_clips=num_clips,
        num_frames_per_clip=num_frames_per_clip,
        sampling_range_start=sampling_range_start,
        sampling_range_end=1000,
    )
    last_clip_start_manual = clips_manual.pts_seconds[:, 0].max()

    assert last_clip_start_manual - last_clip_start_default > 0.3


@pytest.mark.parametrize(
    "sampler",
    (
        partial(
            clips_at_regular_indices,
            num_clips=5,
            sampling_range_start=-30,
        ),
        partial(
            clips_at_regular_timestamps,
            seconds_between_clip_starts=0.02,
            sampling_range_start=12.3,
        ),
    ),
)
def test_sampling_range_default_regular_sampler(sampler):
    # For a regular sampler, checking the default behavior of sampling_range_end
    # is slightly easier: we can assert that the last frame of the last clip
    # *is* the last frame of the video.
    # Note that this doesn't always happen. It depends on specific values passed
    # for num_clips / seconds_between_clip_starts, and where the sampling range
    # starts. We just need to assert that it *can* happen.

    decoder = VideoDecoder(NASA_VIDEO.path)
    last_frame = decoder.get_frame_at(len(decoder) - 1)

    clips = sampler(decoder, num_frames_per_clip=5, policy="error")

    last_clip = clips[-1]
    assert last_clip.pts_seconds[-1] == last_frame.pts_seconds
    assert last_clip.pts_seconds[-2] != last_frame.pts_seconds


@pytest.mark.parametrize(
    "sampler",
    (
        partial(
            clips_at_random_indices, sampling_range_start=-1, sampling_range_end=1000
        ),
        partial(
            clips_at_regular_indices, sampling_range_start=-1, sampling_range_end=1000
        ),
        # Note: the hard-coded value of sampling_range_start=13 is because we know
        # the NASA_VIDEO is ~13.01s seconds long. We just need to clip to start
        # on, or close to the last frame.
        partial(
            clips_at_regular_timestamps,
            seconds_between_clip_starts=0.1,
            sampling_range_start=13,
            sampling_range_end=1000,
        ),
        partial(
            clips_at_random_timestamps,
            sampling_range_start=13,
            sampling_range_end=1000,
        ),
    ),
)
def test_sampling_range_error_policy(sampler):
    decoder = VideoDecoder(NASA_VIDEO.path)
    with pytest.raises(ValueError, match="beyond the number of frames"):
        sampler(
            decoder,
            num_frames_per_clip=10,
            policy="error",
        )


@pytest.mark.parametrize(
    "sampler", (clips_at_random_indices, clips_at_random_timestamps)
)
def test_random_sampler_randomness(sampler):
    decoder = VideoDecoder(NASA_VIDEO.path)
    num_clips = 5

    builtin_random_state_start = random.getstate()

    torch.manual_seed(0)
    clips_1 = sampler(decoder, num_clips=num_clips)

    # Assert the clip starts aren't sorted, to make sure we haven't messed up
    # the implementation. (This may fail if we're unlucky, but we hard-coded a
    # seed, so it will always pass.)
    clip_starts = clips_1.pts_seconds[:, 0].tolist()
    assert sorted(clip_starts) != clip_starts

    # Call the same sampler again with the same seed, expect same results
    torch.manual_seed(0)
    clips_2 = sampler(decoder, num_clips=num_clips)

    for clip_1, clip_2 in zip(clips_1, clips_2):
        assert_frames_equal(clip_1.data, clip_2.data)
        torch.testing.assert_close(
            clip_1.pts_seconds, clip_2.pts_seconds, rtol=0, atol=0
        )
        torch.testing.assert_close(
            clip_1.duration_seconds, clip_2.duration_seconds, rtol=0, atol=0
        )

    # Call with a different seed, expect different results
    torch.manual_seed(1)
    clips_3 = sampler(decoder, num_clips=num_clips)
    with pytest.raises(AssertionError, match="low psnr"):
        assert_frames_equal(clips_1[0].data, clips_3[0].data)

    # Make sure we didn't alter the builtin Python RNG
    builtin_random_state_end = random.getstate()
    assert builtin_random_state_start == builtin_random_state_end


@pytest.mark.parametrize(
    "num_clips, sampling_range_size",
    (
        # Ask for 10 clips while the sampling range is 10 frames wide
        # expect 10 clips with 10 unique starting points.
        (10, 10),
        # Ask for 50 clips while the sampling range is only 10 frames wide
        # expect 50 clips with only 10 unique starting points.
        (50, 10),
    ),
)
def test_sample_at_regular_indices_num_clips_large(num_clips, sampling_range_size):
    # Test for expected behavior described in Note [num clips larger than sampling range]
    decoder = VideoDecoder(NASA_VIDEO.path)
    clips = clips_at_regular_indices(
        decoder,
        num_clips=num_clips,
        sampling_range_start=0,
        sampling_range_end=sampling_range_size,  # because sampling_range_start=0
    )

    assert len(clips) == num_clips

    clip_starts_seconds = clips.pts_seconds[:, 0]
    assert len(torch.unique(clip_starts_seconds)) == sampling_range_size

    # Assert clips starts are ordered, i.e. the start indices don't just "wrap
    # around". They're duplicated *and* ordered.
    assert (clip_starts_seconds.diff() >= 0).all()


@pytest.mark.parametrize(
    "sampler",
    (
        clips_at_random_indices,
        clips_at_regular_indices,
        partial(clips_at_regular_timestamps, seconds_between_clip_starts=1),
        clips_at_random_timestamps,
    ),
)
def test_sampler_errors(sampler):
    decoder = VideoDecoder(NASA_VIDEO.path)
    with pytest.raises(
        ValueError, match=re.escape("num_frames_per_clip (0) must be strictly positive")
    ):
        sampler(decoder, num_frames_per_clip=0)

    with pytest.raises(ValueError, match="Invalid policy"):
        sampler(decoder, policy="BAD")

    with pytest.raises(
        ValueError, match=re.escape("sampling_range_start (1000) must be smaller than")
    ):
        sampler(decoder, sampling_range_start=1000)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "sampling_range_start (4) must be smaller than sampling_range_end"
        ),
    ):
        sampler(decoder, sampling_range_start=4, sampling_range_end=4)


@pytest.mark.parametrize("sampler", (clips_at_random_indices, clips_at_regular_indices))
def test_index_based_samplers_errors(sampler):
    decoder = VideoDecoder(NASA_VIDEO.path)
    with pytest.raises(ValueError, match=re.escape("num_clips (0) must be > 0")):
        sampler(decoder, num_clips=0)

    with pytest.raises(
        ValueError,
        match=re.escape("num_indices_between_frames (0) must be strictly positive"),
    ):
        sampler(decoder, num_indices_between_frames=0)

    with pytest.raises(
        ValueError, match=re.escape("sampling_range_start (290) must be smaller than")
    ):
        sampler(decoder, sampling_range_start=-100, sampling_range_end=-100)

    with pytest.raises(
        ValueError, match="We determined that sampling_range_end should"
    ):
        sampler(
            decoder,
            num_frames_per_clip=10,
            sampling_range_start=len(decoder) - 1,
            sampling_range_end=None,
        )


@pytest.mark.parametrize(
    "sampler",
    (
        clips_at_random_timestamps,
        partial(clips_at_regular_timestamps, seconds_between_clip_starts=1),
    ),
)
def test_time_based_sampler_errors(sampler):
    decoder = VideoDecoder(NASA_VIDEO.path)

    with pytest.raises(
        ValueError, match=re.escape("sampling_range_start (-1) must be at least 0.0")
    ):
        sampler(decoder, sampling_range_start=-1)

    with pytest.raises(
        ValueError, match=re.escape("sampling_range_end (-1) must be at least 0.0")
    ):
        sampler(decoder, sampling_range_end=-1)

    if sampler is clips_at_random_timestamps:
        with pytest.raises(
            ValueError,
            match=re.escape("num_clips (0) must be > 0"),
        ):
            sampler(decoder, num_clips=0)
    else:
        with pytest.raises(
            ValueError, match=re.escape("seconds_between_clip_starts (-1) must be > 0")
        ):
            sampler(decoder, seconds_between_clip_starts=-1)

    @contextlib.contextmanager
    def restore_metadata():
        # Context manager helper that restores the decoder's metadata to its
        # original state upon exit.
        try:
            original_metadata = copy(decoder.metadata)
            yield
        finally:
            decoder.metadata = original_metadata

    with restore_metadata():
        decoder.metadata.end_stream_seconds_from_content = None
        decoder.metadata.duration_seconds_from_header = None
        decoder.metadata.num_frames_from_header = (
            None  # Set to none to prevent fallback calculation
        )
        with pytest.raises(
            ValueError, match="Could not infer stream end from video metadata"
        ):
            sampler(decoder)

    with restore_metadata():
        decoder.metadata.end_stream_seconds_from_content = None
        decoder.metadata.average_fps_from_header = None
        with pytest.raises(ValueError, match="Could not infer average fps"):
            sampler(decoder)


@pytest.mark.parametrize(
    "clip_start_indices, num_indices_between_frames, policy, expected_all_clips_indices",
    (
        (
            [0, 1, 2],  # clip_start_indices
            1,  # num_indices_between_frames
            "repeat_last",  # policy
            # expected_all_clips_indices =
            [0, 1, 2, 3, 4] + [1, 2, 3, 4, 4] + [2, 3, 4, 4, 4],
        ),
        # Same as above but with num_indices_between_frames=2
        (
            [0, 1, 2],  # clip_start_indices
            2,  # num_indices_between_frames
            "repeat_last",  # policy
            # expected_all_clips_indices =
            [0, 2, 4, 4, 4] + [1, 3, 3, 3, 3] + [2, 4, 4, 4, 4],
        ),
        # Same tests as above, for wrap policy
        (
            [0, 1, 2],  # clip_start_indices
            1,  # num_indices_between_frames
            "wrap",  # policy
            # expected_all_clips_indices =
            [0, 1, 2, 3, 4] + [1, 2, 3, 4, 1] + [2, 3, 4, 2, 3],
        ),
        (
            [0, 1, 2],  # clip_start_indices
            2,  # num_indices_between_frames
            "wrap",  # policy
            # expected_all_clips_indices =
            [0, 2, 4, 0, 2] + [1, 3, 1, 3, 1] + [2, 4, 2, 4, 2],
        ),
    ),
)
def test_build_all_clips_indices(
    clip_start_indices, num_indices_between_frames, policy, expected_all_clips_indices
):
    NUM_FRAMES_PER_CLIP = 5
    all_clips_indices = _build_all_clips_indices(
        clip_start_indices=clip_start_indices,
        num_frames_per_clip=NUM_FRAMES_PER_CLIP,
        num_indices_between_frames=num_indices_between_frames,
        num_frames_in_video=5,
        policy_fun=_POLICY_FUNCTIONS[policy],
    )

    assert isinstance(all_clips_indices, list)
    assert all(isinstance(index, int) for index in all_clips_indices)
    assert len(all_clips_indices) == len(clip_start_indices) * NUM_FRAMES_PER_CLIP
    assert all_clips_indices == expected_all_clips_indices


@pytest.mark.parametrize(
    "clip_start_seconds, seconds_between_frames, policy, expected_all_clips_timestamps",
    (
        (
            [0, 1, 2],  # clip_start_seconds
            1.5,  # seconds_between_frames
            "repeat_last",  # policy
            # expected_all_clips_seconds =
            [0, 1.5, 3, 4.5, 4.5] + [1, 2.5, 4, 4, 4] + [2, 3.5, 3.5, 3.5, 3.5],
            # Note how 5 isn't in the last clip, as it's not a seekable pts
            # since we set end_stream_seconds=5
        ),
        # Same as above with wrap policy
        (
            [0, 1, 2],  # clip_start_seconds
            1.5,  # seconds_between_frames
            "wrap",  # policy
            # expected_all_clips_seconds =
            [0, 1.5, 3, 4.5, 0] + [1, 2.5, 4, 1, 2.5] + [2, 3.5, 2, 3.5, 2],
        ),
    ),
)
def test_build_all_clips_timestamps(
    clip_start_seconds, seconds_between_frames, policy, expected_all_clips_timestamps
):
    NUM_FRAMES_PER_CLIP = 5
    all_clips_timestamps = _build_all_clips_timestamps(
        clip_start_seconds=clip_start_seconds,
        num_frames_per_clip=NUM_FRAMES_PER_CLIP,
        seconds_between_frames=seconds_between_frames,
        end_stream_seconds=5.0,
        policy_fun=_POLICY_FUNCTIONS[policy],
    )

    assert isinstance(all_clips_timestamps, list)
    assert all(isinstance(timestamp, float) for timestamp in all_clips_timestamps)
    assert len(all_clips_timestamps) == len(clip_start_seconds) * NUM_FRAMES_PER_CLIP
    assert all_clips_timestamps == expected_all_clips_timestamps


@pytest.mark.parametrize("policy", ("repeat_last", "wrap", "error"))
def test_floating_point_precision_in_clips_at_regular_timestamps(policy):
    # Test that floating point precision errors in torch.arange do not return empty clips.
    # Using 1/3 would cause arange to include sampling_range_end, which gets filtered out
    # in _build_all_clips_timestamps, leaving clips with no frames.
    # The fix rounds seconds_between_clip_starts to prevent this.
    seconds_between_clip_starts = 1 / 3 - 1e-9

    decoder = VideoDecoder(H265_10BITS.path)  # Video is 1 second long
    # Set sampling range so that last clip will have frame timestamp ≈ end_stream_seconds
    sampling_range_start = 0
    sampling_range_end = decoder.metadata.end_stream_seconds
    seconds_between_frames = 1
    num_frames_per_clip = 1

    clips = clips_at_regular_timestamps(
        decoder,
        seconds_between_clip_starts=seconds_between_clip_starts,
        sampling_range_start=sampling_range_start,
        sampling_range_end=sampling_range_end,
        num_frames_per_clip=num_frames_per_clip,
        seconds_between_frames=seconds_between_frames,
        policy=policy,
    )

    # Ensure frame PTS can be decoded
    for clip in clips:
        frames = decoder.get_frames_played_at(seconds=clip.pts_seconds.tolist())
        assert isinstance(frames, FrameBatch)
        assert frames.data.shape[0] == len(clip.pts_seconds)
        assert len(clip.pts_seconds) == num_frames_per_clip
