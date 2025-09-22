from typing import Literal, Optional

import torch

from torchcodec import FrameBatch
from torchcodec.samplers._common import (
    _FRAMEBATCH_RETURN_DOCS,
    _POLICY_FUNCTION_TYPE,
    _POLICY_FUNCTIONS,
    _reshape_4d_framebatch_into_5d,
    _validate_common_params,
)


def _validate_params_time_based(
    *,
    decoder,
    num_clips,
    seconds_between_clip_starts,
    seconds_between_frames,
):

    if (num_clips is None and seconds_between_clip_starts is None) or (
        num_clips is not None and seconds_between_clip_starts is not None
    ):
        raise ValueError("This is internal only and should never happen.")

    if seconds_between_clip_starts is not None and seconds_between_clip_starts <= 0:
        raise ValueError(
            f"seconds_between_clip_starts ({seconds_between_clip_starts}) must be > 0"
        )

    if num_clips is not None and num_clips <= 0:
        raise ValueError(f"num_clips ({num_clips}) must be > 0")

    if decoder.metadata.average_fps is None:
        raise ValueError(
            "Could not infer average fps from video metadata. "
            "Try using an index-based sampler instead."
        )

    # Note that metadata.begin_stream_seconds is a property that will always yield a valid
    # value; if it is not present in the actual metadata, the metadata object will return 0.
    # Hence, we do not test for it here and only test metadata.end_stream_seconds.
    if decoder.metadata.end_stream_seconds is None:
        raise ValueError(
            "Could not infer stream end from video metadata. "
            "Try using an index-based sampler instead."
        )

    average_frame_duration_seconds = 1 / decoder.metadata.average_fps
    if seconds_between_frames is None:
        seconds_between_frames = average_frame_duration_seconds
    elif seconds_between_frames <= 0:
        raise ValueError(
            f"seconds_between_clip_starts ({seconds_between_clip_starts}) must be > 0, got"
        )

    return seconds_between_frames


def _validate_sampling_range_time_based(
    *,
    num_frames_per_clip,
    seconds_between_frames,
    sampling_range_start,
    sampling_range_end,
    begin_stream_seconds,
    end_stream_seconds,
):

    if sampling_range_start is None:
        sampling_range_start = begin_stream_seconds
    else:
        if sampling_range_start < begin_stream_seconds:
            raise ValueError(
                f"sampling_range_start ({sampling_range_start}) must be at least {begin_stream_seconds}"
            )
        if sampling_range_start >= end_stream_seconds:
            raise ValueError(
                f"sampling_range_start ({sampling_range_start}) must be smaller than {end_stream_seconds}"
            )

    if sampling_range_end is None:
        # We allow a clip to start anywhere within
        # [sampling_range_start, sampling_range_end)
        # When sampling_range_end is None, we want to automatically set it to
        # the largest possible value such that the sampled frames in any clip
        # are within the bounds of the video duration (in other words, we don't
        # want to have to resort to the `policy`).
        # I.e. we want to guarantee that for all frames in any clip we have
        # pts < end_stream_seconds.
        #
        # The frames of a clip will be sampled at the following pts:
        # clip_timestamps = [
        #  clip_start + 0 * seconds_between_frames,
        #  clip_start + 1 * seconds_between_frames,
        #  clip_start + 2 * seconds_between_frames,
        #  ...
        #  clip_start + (num_frames_per_clip - 1) * seconds_between_frames,
        # ]
        # To guarantee that any such value is < end_stream_seconds, we only need
        # to guarantee that
        # clip_start < end_stream_seconds - (num_frames_per_clip - 1) * seconds_between_frames
        #
        # So that's the value of sampling_range_end we want to use.
        sampling_range_end = (
            end_stream_seconds - (num_frames_per_clip - 1) * seconds_between_frames
        )
    elif sampling_range_end <= begin_stream_seconds:
        raise ValueError(
            f"sampling_range_end ({sampling_range_end}) must be at least {begin_stream_seconds}"
        )

    if sampling_range_start >= sampling_range_end:
        raise ValueError(
            f"sampling_range_start ({sampling_range_start}) must be smaller than sampling_range_end ({sampling_range_end})"
        )

    sampling_range_end = min(sampling_range_end, end_stream_seconds)

    return sampling_range_start, sampling_range_end


def _build_all_clips_timestamps(
    *,
    clip_start_seconds: torch.Tensor,  # 1D float tensor
    num_frames_per_clip: int,
    seconds_between_frames: float,
    end_stream_seconds: float,
    policy_fun: _POLICY_FUNCTION_TYPE,
) -> list[float]:

    all_clips_timestamps: list[float] = []
    for start_seconds in clip_start_seconds:
        clip_timestamps = [
            timestamp
            for i in range(num_frames_per_clip)
            if (timestamp := start_seconds + i * seconds_between_frames)
            < end_stream_seconds
        ]

        if len(clip_timestamps) < num_frames_per_clip:
            clip_timestamps = policy_fun(clip_timestamps, num_frames_per_clip)
        all_clips_timestamps += clip_timestamps

    return all_clips_timestamps


def _generic_time_based_sampler(
    kind: Literal["random", "regular"],
    decoder,
    *,
    num_clips: Optional[int],  # mutually exclusive with seconds_between_clip_starts
    seconds_between_clip_starts: Optional[float],
    num_frames_per_clip: int,
    seconds_between_frames: Optional[float],
    # None means "begining", which may not always be 0
    sampling_range_start: Optional[float],
    sampling_range_end: Optional[float],  # interval is [start, end).
    policy: Literal["repeat_last", "wrap", "error"] = "repeat_last",
) -> FrameBatch:
    # Note: *everywhere*, sampling_range_end denotes the upper bound of where a
    # clip can start. This is an *open* upper bound, i.e. we will make sure no
    # clip starts exactly at (or above) sampling_range_end.

    _validate_common_params(
        decoder=decoder,
        num_frames_per_clip=num_frames_per_clip,
        policy=policy,
    )

    seconds_between_frames = _validate_params_time_based(
        decoder=decoder,
        num_clips=num_clips,
        seconds_between_clip_starts=seconds_between_clip_starts,
        seconds_between_frames=seconds_between_frames,
    )

    sampling_range_start, sampling_range_end = _validate_sampling_range_time_based(
        num_frames_per_clip=num_frames_per_clip,
        seconds_between_frames=seconds_between_frames,
        sampling_range_start=sampling_range_start,
        sampling_range_end=sampling_range_end,
        begin_stream_seconds=decoder.metadata.begin_stream_seconds,
        end_stream_seconds=decoder.metadata.end_stream_seconds,
    )

    if kind == "random":
        assert num_clips is not None  # appease type-checker
        sampling_range_width = sampling_range_end - sampling_range_start
        # torch.rand() returns in [0, 1)
        # which ensures all clip starts are < sampling_range_end
        clip_start_seconds = (
            torch.rand(num_clips) * sampling_range_width + sampling_range_start
        )
    else:
        assert seconds_between_clip_starts is not None  # appease type-checker
        clip_start_seconds = torch.arange(
            sampling_range_start,
            sampling_range_end,  # excluded
            seconds_between_clip_starts,
        )
        # As mentioned in the docs, torch.arange may return values
        # equal to or above `end` because of floating precision errors.
        # Here, we manually ensure all values are strictly lower than `sample_range_end`
        if clip_start_seconds[-1] >= sampling_range_end:
            clip_start_seconds = clip_start_seconds[
                clip_start_seconds < sampling_range_end
            ]

        num_clips = len(clip_start_seconds)

    all_clips_timestamps = _build_all_clips_timestamps(
        clip_start_seconds=clip_start_seconds,
        num_frames_per_clip=num_frames_per_clip,
        seconds_between_frames=seconds_between_frames,
        end_stream_seconds=decoder.metadata.end_stream_seconds,
        policy_fun=_POLICY_FUNCTIONS[policy],
    )

    frames = decoder.get_frames_played_at(seconds=all_clips_timestamps)
    return _reshape_4d_framebatch_into_5d(
        frames=frames,
        num_clips=num_clips,
        num_frames_per_clip=num_frames_per_clip,
    )


def clips_at_random_timestamps(
    decoder,
    *,
    num_clips: int = 1,
    num_frames_per_clip: int = 1,
    seconds_between_frames: Optional[float] = None,
    # None means "begining", which may not always be 0
    sampling_range_start: Optional[float] = None,
    sampling_range_end: Optional[float] = None,  # interval is [start, end).
    policy: Literal["repeat_last", "wrap", "error"] = "repeat_last",
) -> FrameBatch:
    # See docstring below
    torch._C._log_api_usage_once("torchcodec.samplers.clips_at_random_timestamps")
    return _generic_time_based_sampler(
        kind="random",
        decoder=decoder,
        num_clips=num_clips,
        seconds_between_clip_starts=None,
        num_frames_per_clip=num_frames_per_clip,
        seconds_between_frames=seconds_between_frames,
        sampling_range_start=sampling_range_start,
        sampling_range_end=sampling_range_end,
        policy=policy,
    )


def clips_at_regular_timestamps(
    decoder,
    *,
    seconds_between_clip_starts: float,
    num_frames_per_clip: int = 1,
    seconds_between_frames: Optional[float] = None,
    # None means "begining", which may not always be 0
    sampling_range_start: Optional[float] = None,
    sampling_range_end: Optional[float] = None,  # interval is [start, end).
    policy: Literal["repeat_last", "wrap", "error"] = "repeat_last",
) -> FrameBatch:
    # See docstring below
    torch._C._log_api_usage_once("torchcodec.samplers.clips_at_regular_timestamps")
    return _generic_time_based_sampler(
        kind="regular",
        decoder=decoder,
        num_clips=None,
        seconds_between_clip_starts=seconds_between_clip_starts,
        num_frames_per_clip=num_frames_per_clip,
        seconds_between_frames=seconds_between_frames,
        sampling_range_start=sampling_range_start,
        sampling_range_end=sampling_range_end,
        policy=policy,
    )


_COMMON_DOCS = """
    {maybe_note}

    Args:
        decoder (VideoDecoder): The :class:`~torchcodec.decoders.VideoDecoder`
            instance to sample clips from.
        {num_clips_or_seconds_between_clip_starts}
        num_frames_per_clip (int, optional): The number of frames per clips. Default: 1.
        seconds_between_frames (float or None, optional): The time (in seconds)
            between each frame within a clip. More accurately, this defines the
            time between the *frame sampling point*, i.e. the timestamps at
            which we sample the frames. Because frames span intervals in time ,
            the resulting start of frames within a clip may not be exactly
            spaced by ``seconds_between_frames`` - but on average, they will be.
            Default is None, which is set to the average frame duration
            (``1/average_fps``).
        sampling_range_start (float or None, optional): The start of the
            sampling range, which defines the first timestamp (in seconds) that
            a clip may *start* at. Default: None, which corresponds to the start
            of the video. (Note: some videos start at negative values, which is
            why the default is not 0).
        sampling_range_end (float or None, optional): The end of the sampling
            range, which defines the last timestamp (in seconds) that a clip may
            *start* at. This value is exclusive, i.e. a clip may only start within
            [``sampling_range_start``, ``sampling_range_end``). If None
            (default), the value is set automatically such that the clips never
            span beyond the end of the video, i.e. it is set to
            ``end_video_seconds - (num_frames_per_clip - 1) *
            seconds_between_frames``. When a clip spans beyond the end of the
            video, the ``policy`` parameter defines how to construct such clip.
        policy (str, optional): Defines how to construct clips that span beyond
            the end of the video. This is best described with an example:
            assuming the last valid (seekable) timestamp in a video is 10.9, and
            a clip was sampled to start at timestamp 10.5, with
            ``num_frames_per_clip=5`` and ``seconds_between_frames=0.2``, the
            sampling timestamps of the frames in the clip are supposed to be
            [10.5, 10.7, 10.9, 11.1, 11.2]. But 11.1 and 11.2 are invalid
            timestamps, so the ``policy`` parameter defines how to replace those
            frames, with valid sampling timestamps:

            - "repeat_last": repeats the last valid frame of the clip. We would
              get frames sampled at timestamps [10.5, 10.7, 10.9, 10.9, 10.9].
            - "wrap": wraps around to the beginning of the clip. We would get
              frames sampled at timestamps [10.5, 10.7, 10.9, 10.5, 10.7].
            - "error": raises an error.

            Default is "repeat_last". Note that when ``sampling_range_end=None``
            (default), this policy parameter is unlikely to be relevant.

    {return_docs}
"""


_NUM_CLIPS_DOCS = """
        num_clips (int, optional): The number of clips to return. Default: 1.
"""
clips_at_random_timestamps.__doc__ = f"""Sample :term:`clips` at random timestamps.
{_COMMON_DOCS.format(maybe_note="", num_clips_or_seconds_between_clip_starts=_NUM_CLIPS_DOCS, return_docs=_FRAMEBATCH_RETURN_DOCS)}
"""


_SECONDS_BETWEEN_CLIP_STARTS = """
        seconds_between_clip_starts (float): The space (in seconds) between each
            clip start.
"""

_NOTE_DOCS = """
    .. note::
        For consistency with existing sampling APIs (such as torchvision), this
        sampler takes a ``seconds_between_clip_starts`` parameter instead of
        ``num_clips``. If you find that supporting ``num_clips`` would be
        useful, please let us know by `opening a feature request
        <https://github.com/pytorch/torchcodec/issues?q=is:open+is:issue>`_.
"""
clips_at_regular_timestamps.__doc__ = f"""Sample :term:`clips` at regular (equally-spaced) timestamps.
{_COMMON_DOCS.format(maybe_note=_NOTE_DOCS, num_clips_or_seconds_between_clip_starts=_SECONDS_BETWEEN_CLIP_STARTS, return_docs=_FRAMEBATCH_RETURN_DOCS)}
"""
