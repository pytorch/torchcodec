# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import pytest
import torch

from .utils import H265_VIDEO, NASA_VIDEO


if os.environ.get("IN_FBCODE_TORCHCODEC") != "1":
    pytest.skip(
        "The uniform decode operator is only built in fbcode.",
        allow_module_level=True,
    )


torch.ops.load_library("//pytorch/torchcodec/fb:uniform_decode_ops")


@torch.jit.script
def _decode_video_uniform(
    encoded_video: torch.Tensor,
    requested_frames: int,
    num_threads: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, int, bool]:
    return torch.ops.torchcodec_fb.decode_video_uniform(
        encoded_video,
        requested_frames,
        num_threads,
    )


def _read_video(path: str) -> torch.Tensor:
    with open(path, "rb") as video_file:
        return torch.frombuffer(bytearray(video_file.read()), dtype=torch.uint8)


def test_uniform_sampling_matches_numpy_positive_floor_contract() -> None:
    frames, indices, pts, fps, total_frames, valid = _decode_video_uniform(
        _read_video(str(NASA_VIDEO.path)),
        8,
        2,
    )

    assert valid
    assert frames.shape == (8, 3, 270, 480)
    assert indices.tolist() == [0, 55, 111, 166, 222, 277, 333, 389]
    assert pts.shape == (8,)
    assert abs(fps - 29.97002997002997) < 1e-6
    assert total_frames == 390


def test_decodes_hevc() -> None:
    frames, indices, _, _, total_frames, valid = _decode_video_uniform(
        _read_video(str(H265_VIDEO.path)),
        8,
    )

    assert valid
    assert frames.shape == (8, 3, 128, 128)
    assert indices.tolist() == [0, 1, 2, 3, 5, 6, 7, 9]
    assert total_frames == 10


def test_corrupt_video_returns_status_instead_of_throwing() -> None:
    frames, indices, pts, fps, total_frames, valid = _decode_video_uniform(
        torch.tensor([1, 2, 3], dtype=torch.uint8),
        8,
    )

    assert not valid
    assert frames.shape == (0, 3, 0, 0)
    assert indices.numel() == 0
    assert pts.numel() == 0
    assert fps == 0
    assert total_frames == 0
