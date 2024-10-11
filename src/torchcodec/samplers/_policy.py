from typing import Callable, Union

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
