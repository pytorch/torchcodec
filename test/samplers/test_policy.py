import pytest
from torchcodec.samplers._policy import _POLICY_FUNCTIONS


@pytest.mark.parametrize(
    "policy, frame_indices, expected_frame_indices",
    (
        ("repeat_last", [1, 2, 3], [1, 2, 3, 3, 3]),
        ("repeat_last", [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
        ("wrap", [1, 2, 3], [1, 2, 3, 1, 2]),
        ("wrap", [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
    ),
)
def test_policy(policy, frame_indices, expected_frame_indices):
    policy_fun = _POLICY_FUNCTIONS[policy]
    assert policy_fun(frame_indices, desired_len=5) == expected_frame_indices


def test_error_policy():
    with pytest.raises(ValueError, match="beyond the number of frames"):
        _POLICY_FUNCTIONS["error"]([1, 2, 3], desired_len=5)
