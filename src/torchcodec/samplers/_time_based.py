from typing import List, Literal, Optional

import torch

from torchcodec import Frame, FrameBatch
from torchcodec.decoders import VideoDecoder
from torchcodec.samplers._common import (
    _chunk_list,
    _POLICY_FUNCTION_TYPE,
    _POLICY_FUNCTIONS,
    _to_framebatch,
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
    if (
        decoder.metadata.end_stream_seconds is None
        or decoder.metadata.begin_stream_seconds is None
    ):
        raise ValueError(
            "Could not infer stream end and start from video metadata. "
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
        if sampling_range_start <= begin_stream_seconds:
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


def _decode_all_clips_timestamps(
    decoder: VideoDecoder, all_clips_timestamps: list[float], num_frames_per_clip: int
) -> list[FrameBatch]:
    # This is 99% the same as _decode_all_clips_indices. The only change is the
    # call to .get_frame_displayed_at(pts) instead of .get_frame_at(idx)

    all_clips_timestamps_sorted, argsort = zip(
        *sorted(
            (frame_index, i) for (i, frame_index) in enumerate(all_clips_timestamps)
        )
    )
    previous_decoded_frame = None
    all_decoded_frames = [None] * len(all_clips_timestamps)
    for i, j in enumerate(argsort):
        frame_pts_seconds = all_clips_timestamps_sorted[i]
        if (
            previous_decoded_frame is not None  # then we know i > 0
            and frame_pts_seconds == all_clips_timestamps_sorted[i - 1]
        ):
            # Avoid decoding the same frame twice.
            # Unfortunatly this is unlikely to lead to speed-up as-is: it's
            # pretty unlikely that 2 pts will be the same since pts are float
            # contiguous values. Theoretically the dedup can still happen, but
            # it would be much more efficient to implement it at the frame index
            # level. We should do that once we implement that in C++.
            # See also https://github.com/pytorch/torchcodec/issues/256.
            #
            # IMPORTANT: this is only correct because a copy of the frame will
            # happen within `_to_framebatch` when we call torch.stack.
            # If a copy isn't made, the same underlying memory will be used for
            # the 2 consecutive frames. When we re-write this, we should make
            # sure to explicitly copy the data.
            decoded_frame = previous_decoded_frame
        else:
            decoded_frame = decoder.get_frame_displayed_at(seconds=frame_pts_seconds)
        previous_decoded_frame = decoded_frame
        all_decoded_frames[j] = decoded_frame

    all_clips: list[list[Frame]] = _chunk_list(
        all_decoded_frames, chunk_size=num_frames_per_clip
    )

    return [_to_framebatch(clip) for clip in all_clips]


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
    policy: str = "repeat_last",
) -> List[FrameBatch]:
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

    all_clips_timestamps = _build_all_clips_timestamps(
        clip_start_seconds=clip_start_seconds,
        num_frames_per_clip=num_frames_per_clip,
        seconds_between_frames=seconds_between_frames,
        end_stream_seconds=decoder.metadata.end_stream_seconds,
        policy_fun=_POLICY_FUNCTIONS[policy],
    )

    return _decode_all_clips_timestamps(
        decoder,
        all_clips_timestamps=all_clips_timestamps,
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
    policy: str = "repeat_last",
) -> List[FrameBatch]:
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
    policy: str = "repeat_last",
) -> List[FrameBatch]:
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
