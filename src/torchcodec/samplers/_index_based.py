from typing import Literal, Optional

import torch

from torchcodec import FrameBatch
from torchcodec.decoders import VideoDecoder
from torchcodec.decoders._core import get_frames_at_indices
from torchcodec.samplers._common import (
    _POLICY_FUNCTION_TYPE,
    _POLICY_FUNCTIONS,
    _validate_common_params,
)


def _validate_params_index_based(*, num_clips, num_indices_between_frames):
    if num_clips <= 0:
        raise ValueError(f"num_clips ({num_clips}) must be > 0")

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
) -> FrameBatch:

    _validate_common_params(
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

    frames, pts_seconds, duration_seconds = get_frames_at_indices(
        decoder._decoder,
        stream_index=decoder.stream_index,
        frame_indices=all_clips_indices,
        sort_indices=True,
    )
    last_3_dims = frames.shape[-3:]
    out = FrameBatch(
        data=frames.view(num_clips, num_frames_per_clip, *last_3_dims),
        pts_seconds=pts_seconds.view(num_clips, num_frames_per_clip),
        duration_seconds=duration_seconds.view(num_clips, num_frames_per_clip),
    )
    return [
        FrameBatch(
            out.data[i],
            out.pts_seconds[i],
            out.duration_seconds[i],
        )
        for i in range(out.data.shape[0])
    ]


def clips_at_random_indices(
    decoder: VideoDecoder,
    *,
    num_clips: int = 1,
    num_frames_per_clip: int = 1,
    num_indices_between_frames: int = 1,
    sampling_range_start: int = 0,
    sampling_range_end: Optional[int] = None,  # interval is [start, end).
    policy: Literal["repeat_last", "wrap", "error"] = "repeat_last",
) -> FrameBatch:
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
) -> FrameBatch:

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
