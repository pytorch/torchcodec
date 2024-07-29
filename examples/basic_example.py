# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
==================================================
Basic Example to use TorchCodec to decode a video.
==================================================

A simple example showing how to decode the first few frames of a video  using
the :class:`~torchcodec.decoders.SimpleVideoDecoder` class.
"""

# %%
import inspect
import os

from torchcodec.decoders import SimpleVideoDecoder

# %%
my_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
video_file_path = os.path.dirname(my_path) + "/../test/resources/nasa_13013.mp4"
simple_decoder = SimpleVideoDecoder(video_file_path)

# %%
# You can get the total frame count for the best video stream by calling len().
num_frames = len(simple_decoder)
print(f"{video_file_path=} has {num_frames} frames")

# %%
# You can get the decoded frame by using the subscript operator.
first_frame = simple_decoder[0]
print(f"decoded frame has type {type(first_frame)}")

# %%
# The shape of the decoded frame is (H, W, C) where H and W are the height
# and width of the video frame. C is 3 because we have 3 channels red, green,
# and blue.
print(f"{first_frame.shape=}")

# %%
# The dtype of the decoded frame is ``torch.uint8``.
print(f"{first_frame.dtype=}")

# %%
# Negative indexes are supported.
last_frame = simple_decoder[-1]
print(f"{last_frame.shape=}")

# TODO_BEFORE_RELEASE: add documentation for slices and metadata.
