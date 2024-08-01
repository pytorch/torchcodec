# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


import pytest
import torch
from torchcodec._samplers import (
    DecoderArgs,
    IndexBasedSamplerArgs,
    TimeBasedSamplerArgs,
    VideoArgs,
    VideoClipSampler,
)

from ..utils import assert_tensor_equal, NASA_VIDEO


@pytest.mark.parametrize(
    ("sampler_args"),
    [
        TimeBasedSamplerArgs(
            sampler_type="random", clips_per_video=2, frames_per_clip=4
        ),
        IndexBasedSamplerArgs(
            sampler_type="random", clips_per_video=2, frames_per_clip=4
        ),
        TimeBasedSamplerArgs(
            sampler_type="uniform", clips_per_video=3, frames_per_clip=4
        ),
        IndexBasedSamplerArgs(
            sampler_type="uniform", clips_per_video=3, frames_per_clip=4
        ),
    ],
)
@pytest.mark.parametrize(("device"), [torch.device("cpu"), torch.device("cuda:0")])
def test_sampler(sampler_args, device):
    if device.type == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU not available")

    torch.manual_seed(0)
    desired_width, desired_height = 320, 240
    video_args = VideoArgs(desired_width=desired_width, desired_height=desired_height)
    decoder_args = DecoderArgs(device=device)
    sampler = VideoClipSampler(video_args, sampler_args, decoder_args)
    clips = sampler(NASA_VIDEO.to_tensor())
    assert_tensor_equal(len(clips), sampler_args.clips_per_video)
    clip = clips[0]
    if isinstance(sampler_args, TimeBasedSamplerArgs):
        # Note: Looks like we have an API inconsistency.
        # With time-based sampler, `clip` is a tensor but with index-based
        # samplers `clip` is a list.
        # Below manually convert that list to a tensor for the `.shape` check to
        # be unified, but this block should be removed eventually.
        clip = torch.stack(clip)
    assert clip.shape == (
        sampler_args.frames_per_clip,
        3,
        desired_height,
        desired_width,
    )


if __name__ == "__main__":
    pytest.main()
