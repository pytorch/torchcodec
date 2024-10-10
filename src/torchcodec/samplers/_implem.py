from typing import Callable, List, Literal, Optional, Union

import torch

from torchcodec import Frame, FrameBatch
from torchcodec.decoders import VideoDecoder


def _chunk_list(lst, chunk_size):
    # return list of sublists of length chunk_size
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def _to_framebatch(frames: list[Frame]) -> FrameBatch:
    # IMPORTANT: see other IMPORTANT note in _decode_all_clips_indices and
    # _decode_all_clips_timestamps
    data = torch.stack([frame.data for frame in frames])
    pts_seconds = torch.tensor([frame.pts_seconds for frame in frames])
    duration_seconds = torch.tensor([frame.duration_seconds for frame in frames])
    return FrameBatch(
        data=data, pts_seconds=pts_seconds, duration_seconds=duration_seconds
    )


_LIST_OF_INT_OR_FLOAT = Union[list[int], list[float]]


def _repeat_last_policy(
    values: _LIST_OF_INT_OR_FLOAT, desired_len: int
) -> _LIST_OF_INT_OR_FLOAT:
    # values = [1, 2, 3], desired_len = 5
    # output = [1, 2, 3, 3, 3]
    values += [values[-1]] * (desired_len - len(values))
    return values


def _wrap_policy(
    values: _LIST_OF_INT_OR_FLOAT, desired_len: int
) -> _LIST_OF_INT_OR_FLOAT:
    # values = [1, 2, 3], desired_len = 5
    # output = [1, 2, 3, 1, 2]
    return (values * (desired_len // len(values) + 1))[:desired_len]


def _error_policy(
    frames_indices: _LIST_OF_INT_OR_FLOAT, desired_len: int
) -> _LIST_OF_INT_OR_FLOAT:
    raise ValueError(
        "You set the 'error' policy, and the sampler tried to decode a frame "
        "that is beyond the number of frames in the video. "
        "Try to leave sampling_range_end to its default value?"
    )


_POLICY_FUNCTION_TYPE = Callable[[_LIST_OF_INT_OR_FLOAT, int], _LIST_OF_INT_OR_FLOAT]
_POLICY_FUNCTIONS: dict[str, _POLICY_FUNCTION_TYPE] = {
    "repeat_last": _repeat_last_policy,
    "wrap": _wrap_policy,
    "error": _error_policy,
}


def _validate_params(*, decoder, num_frames_per_clip, policy):
    if len(decoder) < 1:
        raise ValueError(
            f"Decoder must have at least one frame, found {len(decoder)} frames."
        )

    if num_frames_per_clip <= 0:
        raise ValueError(
            f"num_frames_per_clip ({num_frames_per_clip}) must be strictly positive"
        )
    if policy not in _POLICY_FUNCTIONS.keys():
        raise ValueError(
            f"Invalid policy ({policy}). Supported values are {_POLICY_FUNCTIONS.keys()}."
        )


def _validate_params_index_based(*, num_clips, num_indices_between_frames):
    if num_clips <= 0:
        raise ValueError(f"num_clips ({num_clips}) must be strictly positive")

    if num_indices_between_frames <= 0:
        raise ValueError(
            f"num_indices_between_frames ({num_indices_between_frames}) must be strictly positive"
        )


def _validate_sampling_range_index_based(
    *,
    num_indices_between_frames,
    num_frames_per_clip,
    sampling_range_start,
    sampling_range_end,
    num_frames_in_video,
):
    if sampling_range_start < 0:
        sampling_range_start = num_frames_in_video + sampling_range_start

    if sampling_range_start >= num_frames_in_video:
        raise ValueError(
            f"sampling_range_start ({sampling_range_start}) must be smaller than "
            f"the number of frames ({num_frames_in_video})."
        )

    clip_span = _get_clip_span(
        num_indices_between_frames=num_indices_between_frames,
        num_frames_per_clip=num_frames_per_clip,
    )

    if sampling_range_end is None:
        sampling_range_end = max(num_frames_in_video - clip_span + 1, 1)
        if sampling_range_start >= sampling_range_end:
            raise ValueError(
                f"We determined that sampling_range_end should be {sampling_range_end}, "
                "but it is smaller than or equal to sampling_range_start "
                f"({sampling_range_start})."
            )
    else:
        if sampling_range_end < 0:
            # Support negative values so that -1 means last frame.
            sampling_range_end = num_frames_in_video + sampling_range_end
        sampling_range_end = min(sampling_range_end, num_frames_in_video)
        if sampling_range_start >= sampling_range_end:
            raise ValueError(
                f"sampling_range_start ({sampling_range_start}) must be smaller than "
                f"sampling_range_end ({sampling_range_end})."
            )

    return sampling_range_start, sampling_range_end


def _get_clip_span(*, num_indices_between_frames, num_frames_per_clip):
    """Return the span of a clip, i.e. the number of frames (or indices)
    between the first and last frame in the clip, both included.

    This isn't the same as the number of frames in a clip!
    Example: f means a frame in the clip, x means a frame excluded from the clip
    num_frames_per_clip = 4
    num_indices_between_frames = 1, clip = ffff      , span = 4
    num_indices_between_frames = 2, clip = fxfxfxf   , span = 7
    num_indices_between_frames = 3, clip = fxxfxxfxxf, span = 10
    """
    return num_indices_between_frames * (num_frames_per_clip - 1) + 1


def _build_all_clips_indices(
    *,
    clip_start_indices: torch.Tensor,  # 1D int tensor
    num_frames_per_clip: int,
    num_indices_between_frames: int,
    num_frames_in_video: int,
    policy_fun: _POLICY_FUNCTION_TYPE,
) -> list[int]:
    # From the clip_start_indices [f_00, f_10, f_20, ...]
    # and from the rest of the parameters, return the list of all the frame
    # indices that make up all the clips.
    # I.e. the output is [f_00, f_01, f_02, f_03, f_10, f_11, f_12, f_13, ...]
    # where f_01 is the index of frame 1 in clip 0.
    #
    # All clips in the output are of length num_frames_per_clip (=4 in example
    # above). When the frame indices go beyond num_frames_in_video, we force the
    # frame indices back to valid values by applying the user's policy (wrap,
    # repeat, etc.).
    all_clips_indices: list[int] = []

    clip_span = _get_clip_span(
        num_indices_between_frames=num_indices_between_frames,
        num_frames_per_clip=num_frames_per_clip,
    )

    for start_index in clip_start_indices:
        frame_index_upper_bound = min(start_index + clip_span, num_frames_in_video)
        frame_indices = list(
            range(start_index, frame_index_upper_bound, num_indices_between_frames)
        )
        if len(frame_indices) < num_frames_per_clip:
            frame_indices = policy_fun(frame_indices, num_frames_per_clip)  # type: ignore[assignment]
        all_clips_indices += frame_indices
    return all_clips_indices


def _decode_all_clips_indices(
    decoder: VideoDecoder, all_clips_indices: list[int], num_frames_per_clip: int
) -> list[FrameBatch]:
    # This takes the list of all the frames to decode (in arbitrary order),
    # decode all the frames, and then packs them into clips of length
    # num_frames_per_clip.
    #
    # To avoid backwards seeks (which are slow), we:
    # - sort all the frame indices to be decoded
    # - dedup them
    # - decode all unique frames in sorted order
    # - re-assemble the decoded frames back to their original order
    #
    # TODO: Write this in C++ so we can avoid the copies that happen in `_to_framebatch`

    all_clips_indices_sorted, argsort = zip(
        *sorted((frame_index, i) for (i, frame_index) in enumerate(all_clips_indices))
    )
    previous_decoded_frame = None
    all_decoded_frames = [None] * len(all_clips_indices)
    for i, j in enumerate(argsort):
        frame_index = all_clips_indices_sorted[i]
        if (
            previous_decoded_frame is not None  # then we know i > 0
            and frame_index == all_clips_indices_sorted[i - 1]
        ):
            # Avoid decoding the same frame twice.
            # IMPORTANT: this is only correct because a copy of the frame will
            # happen within `_to_framebatch` when we call torch.stack.
            # If a copy isn't made, the same underlying memory will be used for
            # the 2 consecutive frames. When we re-write this, we should make
            # sure to explicitly copy the data.
            decoded_frame = previous_decoded_frame
        else:
            decoded_frame = decoder.get_frame_at(index=frame_index)
        previous_decoded_frame = decoded_frame
        all_decoded_frames[j] = decoded_frame

    all_clips: list[list[Frame]] = _chunk_list(
        all_decoded_frames, chunk_size=num_frames_per_clip
    )

    return [_to_framebatch(clip) for clip in all_clips]


def _generic_index_based_sampler(
    kind: Literal["random", "regular"],
    decoder: VideoDecoder,
    *,
    num_clips: int,
    num_frames_per_clip: int,
    num_indices_between_frames: int,
    sampling_range_start: int,
    sampling_range_end: Optional[int],  # interval is [start, end).
    # Important note: sampling_range_end defines the upper bound of where a clip
    # can *start*, not where a clip can end.
    policy: Literal["repeat_last", "wrap", "error"],
) -> List[FrameBatch]:

    _validate_params(
        decoder=decoder,
        num_frames_per_clip=num_frames_per_clip,
        policy=policy,
    )
    _validate_params_index_based(
        num_clips=num_clips,
        num_indices_between_frames=num_indices_between_frames,
    )

    sampling_range_start, sampling_range_end = _validate_sampling_range_index_based(
        num_frames_per_clip=num_frames_per_clip,
        num_indices_between_frames=num_indices_between_frames,
        sampling_range_start=sampling_range_start,
        sampling_range_end=sampling_range_end,
        num_frames_in_video=len(decoder),
    )

    if kind == "random":
        clip_start_indices = torch.randint(
            low=sampling_range_start, high=sampling_range_end, size=(num_clips,)
        )
    else:
        # Note [num clips larger than sampling range]
        # If we ask for more clips than there are frames in the sampling range or
        # in the video, we rely on torch.linspace behavior which will return
        # duplicated indices.
        # E.g. torch.linspace(0, 10, steps=20, dtype=torch.int) returns
        # 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 10
        # Alternatively we could wrap around, but the current behavior is closer to
        # the expected "equally spaced indices" sampling.
        clip_start_indices = torch.linspace(
            sampling_range_start,
            sampling_range_end - 1,
            steps=num_clips,
            dtype=torch.int,
        )

    all_clips_indices = _build_all_clips_indices(
        clip_start_indices=clip_start_indices,
        num_frames_per_clip=num_frames_per_clip,
        num_indices_between_frames=num_indices_between_frames,
        num_frames_in_video=len(decoder),
        policy_fun=_POLICY_FUNCTIONS[policy],
    )
    return _decode_all_clips_indices(
        decoder,
        all_clips_indices=all_clips_indices,
        num_frames_per_clip=num_frames_per_clip,
    )


def clips_at_random_indices(
    decoder: VideoDecoder,
    *,
    num_clips: int = 1,
    num_frames_per_clip: int = 1,
    num_indices_between_frames: int = 1,
    sampling_range_start: int = 0,
    sampling_range_end: Optional[int] = None,  # interval is [start, end).
    policy: Literal["repeat_last", "wrap", "error"] = "repeat_last",
) -> List[FrameBatch]:
    return _generic_index_based_sampler(
        kind="random",
        decoder=decoder,
        num_clips=num_clips,
        num_frames_per_clip=num_frames_per_clip,
        num_indices_between_frames=num_indices_between_frames,
        sampling_range_start=sampling_range_start,
        sampling_range_end=sampling_range_end,
        policy=policy,
    )


def clips_at_regular_indices(
    decoder: VideoDecoder,
    *,
    num_clips: int = 1,
    num_frames_per_clip: int = 1,
    num_indices_between_frames: int = 1,
    sampling_range_start: int = 0,
    sampling_range_end: Optional[int] = None,  # interval is [start, end).
    policy: Literal["repeat_last", "wrap", "error"] = "repeat_last",
) -> List[FrameBatch]:

    return _generic_index_based_sampler(
        kind="regular",
        decoder=decoder,
        num_clips=num_clips,
        num_frames_per_clip=num_frames_per_clip,
        num_indices_between_frames=num_indices_between_frames,
        sampling_range_start=sampling_range_start,
        sampling_range_end=sampling_range_end,
        policy=policy,
    )


def _validate_params_time_based(
    *,
    decoder,
    seconds_between_clip_starts,
    seconds_between_frames,
):
    if seconds_between_clip_starts <= 0:
        raise ValueError(
            f"seconds_between_clip_starts ({seconds_between_clip_starts}) must be > 0"
        )

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
    # Note: *everywhere*, sampling_range_end denotes the upper bound of where a
    # clip can start. This is an *open* upper bound, i.e. we will make sure no
    # clip starts exactly at (or above) sampling_range_end.

    _validate_params(
        decoder=decoder,
        num_frames_per_clip=num_frames_per_clip,
        policy=policy,
    )

    seconds_between_frames = _validate_params_time_based(
        decoder=decoder,
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
