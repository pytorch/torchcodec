# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import importlib

import torch
from memory_profiler import profile
from torchcodec._core import add_video_stream, create_from_file, get_next_frame

torch._dynamo.config.cache_size_limit = 100
torch._dynamo.config.capture_dynamic_output_shape_ops = True


@profile
def torchcodec_create_next(video_file):
    video_decoder = create_from_file(video_file)
    add_video_stream(video_decoder)
    get_next_frame(video_decoder)
    return video_decoder


def get_video_path_str(filename: str) -> str:
    resource = importlib.resources.files(__package__).joinpath(filename)
    with importlib.resources.as_file(resource) as path:
        return str(path)


def main() -> None:
    """Memory leak check and profiling for decoders."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iterations",
        help="Number of times to invoke decoder operations.",
        type=int,
        default=10,
    )
    args = parser.parse_args()

    large_video_path = get_video_path_str("853.mp4")

    # We call the same function several times, and each call will produce memory stats on
    # standard out. We rely on a human looking at the output to see if memory increases
    # on each run.
    for _ in range(args.iterations):
        torchcodec_create_next(large_video_path)


if __name__ == "__main__":
    main()
