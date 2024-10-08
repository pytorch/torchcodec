from typing import Callable, List, Literal, Optional

import torch

from torchcodec import Frame, FrameBatch
from torchcodec.decoders import VideoDecoder


_EPS = 1e-4


def _validate_params(
    *, decoder, num_clips, num_frames_per_clip, num_indices_between_frames, policy
):
    if len(decoder) < 1:
        raise ValueError(
            f"Decoder must have at least one frame, found {len(decoder)} frames."
        )

    if num_clips <= 0:
        raise ValueError(f"num_clips ({num_clips}) must be strictly positive")
    if num_frames_per_clip <= 0:
        raise ValueError(
            f"num_frames_per_clip ({num_frames_per_clip}) must be strictly positive"
        )
    if num_indices_between_frames <= 0:
        raise ValueError(
            f"num_indices_between_frames ({num_indices_between_frames}) must be strictly positive"
        )

    if policy not in _POLICY_FUNCTIONS.keys():
        raise ValueError(
            f"Invalid policy ({policy}). Supported values are {_POLICY_FUNCTIONS.keys()}."
        )


def _validate_sampling_range(
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


def _repeat_last_policy(
    frame_indices: list[int], num_frames_per_clip: int
) -> list[int]:
    # frame_indices = [1, 2, 3], num_frames_per_clip = 5
    # output = [1, 2, 3, 3, 3]
    frame_indices += [frame_indices[-1]] * (num_frames_per_clip - len(frame_indices))
    return frame_indices


def _wrap_policy(frame_indices: list[int], num_frames_per_clip: int) -> list[int]:
    # frame_indices = [1, 2, 3], num_frames_per_clip = 5
    # output = [1, 2, 3, 1, 2]
    return (frame_indices * (num_frames_per_clip // len(frame_indices) + 1))[
        :num_frames_per_clip
    ]


def _error_policy(frames_indices: list[int], num_frames_per_clip: int) -> list[int]:
    raise ValueError(
        "You set the 'error' policy, and the sampler tried to decode a frame "
        "that is beyond the number of frames in the video. "
        "Try to leave sampling_range_end to its default value?"
    )


_POLICY_FUNCTION_TYPE = Callable[[list[int], int], list[int]]
_POLICY_FUNCTIONS: dict[str, _POLICY_FUNCTION_TYPE] = {
    "repeat_last": _repeat_last_policy,
    "wrap": _wrap_policy,
    "error": _error_policy,
}


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
            frame_indices = policy_fun(frame_indices, num_frames_per_clip)
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
    # TODO: Write this in C++ so we can avoid the copies that happen in `to_framebatch`

    def chunk_list(lst, chunk_size):
        # return list of sublists of length chunk_size
        return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

    def to_framebatch(frames: list[Frame]) -> FrameBatch:
        # IMPORTANT: see other IMPORTANT note below
        data = torch.stack([frame.data for frame in frames])
        pts_seconds = torch.tensor([frame.pts_seconds for frame in frames])
        duration_seconds = torch.tensor([frame.duration_seconds for frame in frames])
        return FrameBatch(
            data=data, pts_seconds=pts_seconds, duration_seconds=duration_seconds
        )

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
            # happen within `to_framebatch` when we call torch.stack.
            # If a copy isn't made, the same underlying memory will be used for
            # the 2 consecutive frames. When we re-write this, we should make
            # sure to explicitly copy the data.
            decoded_frame = previous_decoded_frame
        else:
            decoded_frame = decoder.get_frame_at(index=frame_index)
        previous_decoded_frame = decoded_frame
        all_decoded_frames[j] = decoded_frame

    all_clips: list[list[Frame]] = chunk_list(
        all_decoded_frames, chunk_size=num_frames_per_clip
    )

    return [to_framebatch(clip) for clip in all_clips]


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
        num_clips=num_clips,
        num_frames_per_clip=num_frames_per_clip,
        num_indices_between_frames=num_indices_between_frames,
        policy=policy,
    )

    sampling_range_start, sampling_range_end = _validate_sampling_range(
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


def _get_approximate_clip_span_seconds(
    *,
    decoder,
    num_frames_per_clip,
    seconds_between_frames,
):

    # Compute clip span, in seconds. We can only compute an approximate value:
    # we assume the fps are constant. Computing the real value requires
    # accounting for variable fps.

    assert decoder.metadata.average_fps is not None
    average_frame_duration_seconds = 1 / decoder.metadata.average_fps
    if seconds_between_frames is None:
        approximate_clip_span_seconds = (
            num_frames_per_clip * average_frame_duration_seconds
        )
    else:
        # aaa, bbb, ccc, ddd are 4 frames within a clip.
        #
        #      seconds_between_frames
        #              |
        #              v
        #           < ---- >
        #   clip = [aaa....bbb....ccc....ddd]
        #           < ----------------- >
        #                    ^
        #                    |
        #  (num_frames_per_clip - 1) * seconds_between_frames
        #
        # Now to compute the clip span, we need to add the duration of the last
        # frame. The formula is fairly approximate, as we assume fps are
        # constant, and that
        # seconds_between_frames > average_frame_duration_seconds. It's good
        # enough for what we need to do.
        approximate_clip_span_seconds = (
            num_frames_per_clip - 1
        ) * seconds_between_frames + average_frame_duration_seconds

    return approximate_clip_span_seconds


def _validate_sampling_range_time_based(
    *,
    decoder,
    num_frames_per_clip,
    seconds_between_frames,
    sampling_range_start,
    sampling_range_end,
):
    assert decoder.metadata.end_stream_seconds is not None
    if sampling_range_start is None:
        assert decoder.metadata.begin_stream_seconds is not None
        sampling_range_start = decoder.metadata.begin_stream_seconds

    if sampling_range_end is None:
        approximate_clip_span_seconds = _get_approximate_clip_span_seconds(
            decoder=decoder,
            seconds_between_frames=seconds_between_frames,
            num_frames_per_clip=num_frames_per_clip,
        )
        sampling_range_end = (
            decoder.metadata.end_stream_seconds - approximate_clip_span_seconds
        )
    sampling_range_end = min(
        sampling_range_end, decoder.metadata.end_stream_seconds - _EPS
    )

    return sampling_range_start, sampling_range_end


def _build_all_clips_timestamps(
    *,
    decoder: VideoDecoder,
    clip_start_seconds: torch.Tensor,  # 1D float tensor
    num_frames_per_clip: int,
    seconds_between_frames: Optional[float],
    end_video_seconds: float,
    policy_fun: _POLICY_FUNCTION_TYPE,
) -> list[int]:
    all_clips_timestamps: list[float] = []

    approximate_clip_span_seconds = _get_approximate_clip_span_seconds(
        decoder=decoder,
        num_frames_per_clip=num_frames_per_clip,
        seconds_between_frames=seconds_between_frames,
    )

    if seconds_between_frames is None:
        average_frame_duration_seconds = 1 / decoder.metadata.average_fps
        seconds_between_frames = average_frame_duration_seconds
        # TODO: What we're doing above defeats the purpose of having a
        # time-based sampler because we are assuming constant fps. We won't
        # accurately get consecutive frames for variable fps, while this is the
        # desired behavior when seconds_between_frames is None. I think we need
        # an API like next_pts = decoder._get_next_pts(current_pts) that returns
        # the pts of the *next* frame. i.e. if frame i is the one displayed at
        # current_pts, we want the pts of frame i+1.

    for start_seconds in clip_start_seconds:
        frame_pts_upper_bound = min(
            start_seconds + approximate_clip_span_seconds, end_video_seconds - _EPS
        )
        # This is correct when seconds_between_frames is specified by the user,
        # but not quite correct when it's None if fps are variable. See note
        # above.
        frame_pts = torch.arange(
            start_seconds, frame_pts_upper_bound, step=seconds_between_frames
        ).tolist()
        if len(frame_pts) < num_frames_per_clip:
            frame_pts = policy_fun(frame_pts, num_frames_per_clip)
        all_clips_timestamps += frame_pts

    return all_clips_timestamps


def _decode_all_clips_timestamps(
    decoder: VideoDecoder, all_clips_timestamps: list[int], num_frames_per_clip: int
) -> list[FrameBatch]:
    # This is 99% the same as _decode_all_clips_indices. The only change is the
    # call to .get_frame_displayed_at(pts) instead of .get_frame_at(idx)

    def chunk_list(lst, chunk_size):
        # return list of sublists of length chunk_size
        return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

    def to_framebatch(frames: list[Frame]) -> FrameBatch:
        # IMPORTANT: see other IMPORTANT note below
        data = torch.stack([frame.data for frame in frames])
        pts_seconds = torch.tensor([frame.pts_seconds for frame in frames])
        duration_seconds = torch.tensor([frame.duration_seconds for frame in frames])
        return FrameBatch(
            data=data, pts_seconds=pts_seconds, duration_seconds=duration_seconds
        )

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
            # happen within `to_framebatch` when we call torch.stack.
            # If a copy isn't made, the same underlying memory will be used for
            # the 2 consecutive frames. When we re-write this, we should make
            # sure to explicitly copy the data.
            decoded_frame = previous_decoded_frame
        else:
            decoded_frame = decoder.get_frame_displayed_at(seconds=frame_pts_seconds)
        previous_decoded_frame = decoded_frame
        all_decoded_frames[j] = decoded_frame

    all_clips: list[list[Frame]] = chunk_list(
        all_decoded_frames, chunk_size=num_frames_per_clip
    )

    return [to_framebatch(clip) for clip in all_clips]


def clips_at_regular_timestamps(
    decoder,
    *,
    seconds_between_clip_starts: int,  # TODO or its inverse: num_clips_per_seconds?
    num_frames_per_clip: int = 1,
    seconds_between_frames: Optional[float] = None,
    # None means "begining", which may not always be 0
    sampling_range_start: Optional[float] = None,
    sampling_range_end: Optional[float] = None,
    policy: str = "repeat_last",
) -> List[FrameBatch]:

    # TODO: better validation
    assert seconds_between_clip_starts > 0
    assert num_frames_per_clip > 0
    assert seconds_between_frames is None or seconds_between_frames > 0
    assert sampling_range_start is None or sampling_range_start >= 0
    assert sampling_range_end is None or sampling_range_end >= 0

    sampling_range_start, sampling_range_end = _validate_sampling_range_time_based(
        decoder=decoder,
        num_frames_per_clip=num_frames_per_clip,
        seconds_between_frames=seconds_between_frames,
        sampling_range_start=sampling_range_start,
        sampling_range_end=sampling_range_end,
    )

    sampling_range_seconds = sampling_range_end - sampling_range_start
    num_clips = int(round(sampling_range_seconds / seconds_between_clip_starts))

    clip_start_seconds = torch.linspace(
        sampling_range_start,
        sampling_range_end,
        steps=num_clips,
    )

    all_clips_timestamps = _build_all_clips_timestamps(
        decoder=decoder,
        clip_start_seconds=clip_start_seconds,
        num_frames_per_clip=num_frames_per_clip,
        seconds_between_frames=seconds_between_frames,
        end_video_seconds=decoder.metadata.end_stream_seconds,
        policy_fun=_POLICY_FUNCTIONS[policy],
    )

    return _decode_all_clips_timestamps(
        decoder,
        all_clips_timestamps=all_clips_timestamps,
        num_frames_per_clip=num_frames_per_clip,
    )
