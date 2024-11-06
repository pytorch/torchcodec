from typing import Callable, Union

from torchcodec import FrameBatch

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


def _validate_common_params(*, decoder, num_frames_per_clip, policy):
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


def _reshape_4d_framebatch_into_5d(
    *,
    frames: FrameBatch,
    num_clips: int,
    num_frames_per_clip: int,
) -> FrameBatch:
    last_3_dims = frames.data.shape[-3:]
    return FrameBatch(
        data=frames.data.view(num_clips, num_frames_per_clip, *last_3_dims),
        pts_seconds=frames.pts_seconds.view(num_clips, num_frames_per_clip),
        duration_seconds=frames.duration_seconds.view(num_clips, num_frames_per_clip),
    )


_FRAMEBATCH_RETURN_DOCS = """
    Returns:
        FrameBatch:
            The sampled :term:`clips`, as a 5D :class:`~torchcodec.FrameBatch`.
            The shape of the ``data`` field is (``num_clips``,
            ``num_frames_per_clips``, ...) where ... is (H, W, C) or (C, H, W)
            depending on the ``dimension_order`` parameter of
            :class:`~torchcodec.decoders.VideoDecoder`. The shape of the
            ``pts_seconds`` and ``duration_seconds`` fields is (``num_clips``,
            ``num_frames_per_clips``).
"""
