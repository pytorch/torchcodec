from typing import Callable, Union

import torch
from torchcodec import Frame, FrameBatch

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
