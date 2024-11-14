from typing import Literal, Optional

import torch

from torchcodec import FrameBatch
from torchcodec.decoders import VideoDecoder
from torchcodec.samplers._common import (
    _FRAMEBATCH_RETURN_DOCS,
    _POLICY_FUNCTION_TYPE,
    _POLICY_FUNCTIONS,
    _reshape_4d_framebatch_into_5d,
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

    frames = decoder.get_frames_at(indices=all_clips_indices)
    return _reshape_4d_framebatch_into_5d(
        frames=frames,
        num_clips=num_clips,
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
) -> FrameBatch:
    # See docstring below
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
    # See docstring below
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


_COMMON_DOCS = f"""
    Args:
        decoder (VideoDecoder): The :class:`~torchcodec.decoders.VideoDecoder`
            instance to sample clips from.
        num_clips (int, optional): The number of clips to return. Default: 1.
        num_frames_per_clip (int, optional): The number of frames per clips. Default: 1.
        num_indices_between_frames(int, optional): The number of indices between
            the frames *within* a clip. Default: 1, which means frames are
            consecutive. This is sometimes refered-to as "dilation".
        sampling_range_start (int, optional): The start of the sampling range,
            which defines the first index that a clip may *start* at. Default:
            0, i.e. the start of the video.
        sampling_range_end (int or None, optional): The end of the sampling
            range, which defines the last index that a clip may *start* at. This
            value is exclusive, i.e. a clip may only start within
            [``sampling_range_start``, ``sampling_range_end``). If None
            (default), the value is set automatically such that the clips never
            span beyond the end of the video. For example if the last valid
            index in a video is 99 and the clips span 10 frames, this value is
            set to 99 - 10 + 1 = 90. Negative values are accepted and are
            equivalent to ``len(video) - val``. When a clip spans beyond the end
            of the video, the ``policy`` parameter defines how to construct such
            clip.
        policy (str, optional): Defines how to construct clips that span beyond
            the end of the video. This is best described with an example:
            assuming the last valid index in a video is 99, and a clip was
            sampled to start at index 95, with ``num_frames_per_clip=5`` and
            ``num_indices_between_frames=2``, the indices of the frames in the
            clip are supposed to be [95, 97, 99, 101, 103]. But 101 and 103 are
            invalid indices, so the ``policy`` parameter defines how to replace
            those frames, with valid indices:

            - "repeat_last": repeats the last valid frame of the clip. We would
              get [95, 97, 99, 99, 99].
            - "wrap": wraps around to the beginning of the clip. We would get
              [95, 97, 99, 95, 97].
            - "error": raises an error.

            Default is "repeat_last". Note that when ``sampling_range_end=None``
            (default), this policy parameter is unlikely to be relevant.

    {_FRAMEBATCH_RETURN_DOCS}
"""

clips_at_random_indices.__doc__ = f"""Sample :term:`clips` at random indices.
{_COMMON_DOCS}
"""


clips_at_regular_indices.__doc__ = f"""Sample :term:`clips` at regular (equally-spaced) indices.
{_COMMON_DOCS}
"""
