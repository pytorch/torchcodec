# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
========================================
Decoding a video with VideoDecoder
========================================

In this example, we'll learn how to decode a video using the
:class:`~torchcodec.decoders.VideoDecoder` class.
"""

# %%
# First, a bit of boilerplate: we'll download a video from the web, and define a
# plotting utility. You can ignore that part and jump right below to
# :ref:`creating_decoder`.

from typing import Optional
import torch
import requests


# Video source: https://www.pexels.com/video/dog-eating-854132/
# License: CC0. Author: Coverr.
url = "https://videos.pexels.com/video-files/854132/854132-sd_640_360_25fps.mp4"
response = requests.get(url, headers={"User-Agent": ""})
if response.status_code != 200:
    raise RuntimeError(f"Failed to download video. {response.status_code = }.")

raw_video_bytes = response.content


def plot(frames: torch.Tensor, title : Optional[str] = None):
    try:
        from torchvision.utils import make_grid
        from torchvision.transforms.v2.functional import to_pil_image
        import matplotlib.pyplot as plt
    except ImportError:
        print("Cannot plot, please run `pip install torchvision matplotlib`")
        return

    plt.rcParams["savefig.bbox"] = 'tight'
    fig, ax = plt.subplots()
    ax.imshow(to_pil_image(make_grid(frames)))
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    if title is not None:
        ax.set_title(title)
    plt.tight_layout()


# %%
# .. _creating_decoder:
#
# Creating a decoder
# ------------------
#
# We can now create a decoder from the raw (encoded) video bytes. You can of
# course use a local video file and pass the path as input, rather than download
# a video.
from torchcodec.decoders import VideoDecoder

# You can also pass a path to a local file!
decoder = VideoDecoder(raw_video_bytes)

# %%
# The video has not yet been decoded by the decoder, but we already have access to
# some metadata via the ``metadata`` attribute which is a
# :class:`~torchcodec.decoders.VideoStreamMetadata` object.
print(decoder.metadata)

# %%
# Decoding frames by indexing the decoder
# ---------------------------------------

first_frame = decoder[0]  # using a single int index
every_twenty_frame = decoder[0 : -1 : 20]  # using slices

print(f"{first_frame.shape = }")
print(f"{first_frame.dtype = }")
print(f"{every_twenty_frame.shape = }")
print(f"{every_twenty_frame.dtype = }")

# %%
# Indexing the decoder returns the frames as :class:`torch.Tensor` objects.
# By default, the shape of the frames is ``(N, C, H, W)`` where N is the batch
# size C the number of channels, H is the height, and W is the width of the
# frames.  The batch dimension N is only present when we're decoding more than
# one frame. The dimension order can be changed to ``N, H, W, C`` using the
# ``dimension_order`` parameter of
# :class:`~torchcodec.decoders.VideoDecoder`. Frames are always of
# ``torch.uint8`` dtype.
#
# .. note::
#
#     If you need to decode multiple frames, we recommend using the batch
#     methods instead, since they are faster:
#     :meth:`~torchcodec.decoders.VideoDecoder.get_frames_at`,
#     :meth:`~torchcodec.decoders.VideoDecoder.get_frames_in_range`,
#     :meth:`~torchcodec.decoders.VideoDecoder.get_frames_played_at`, and
#     :meth:`~torchcodec.decoders.VideoDecoder.get_frames_played_in_range`. They
#     are described below.


plot(first_frame, "First frame")

# %%
plot(every_twenty_frame, "Every 20 frame")

# %%
# Iterating over frames
# ---------------------
#
# The decoder is a normal iterable object and can be iterated over like so:

for frame in decoder:
    assert (
        isinstance(frame, torch.Tensor)
        and frame.shape == (3, decoder.metadata.height, decoder.metadata.width)
    )

# %%
# Retrieving pts and duration of frames
# -------------------------------------
#
# Indexing the decoder returns pure :class:`torch.Tensor` objects. Sometimes, it
# can be useful to retrieve additional information about the frames, such as
# their :term:`pts` (Presentation Time Stamp), and their duration.
# This can be achieved using the
# :meth:`~torchcodec.decoders.VideoDecoder.get_frame_at` and
# :meth:`~torchcodec.decoders.VideoDecoder.get_frames_at`  methods, which will
# return a :class:`~torchcodec.Frame` and :class:`~torchcodec.FrameBatch`
# objects respectively.

last_frame = decoder.get_frame_at(len(decoder) - 1)
print(f"{type(last_frame) = }")
print(last_frame)

# %%
other_frames = decoder.get_frames_at([10, 0, 50])
print(f"{type(other_frames) = }")
print(other_frames)

# %%
plot(last_frame.data, "Last frame")
plot(other_frames.data, "Other frames")

# %%
# Both :class:`~torchcodec.Frame` and
# :class:`~torchcodec.FrameBatch` have a ``data`` field, which contains
# the decoded tensor data. They also have the ``pts_seconds`` and
# ``duration_seconds`` fields which are single ints for
# :class:`~torchcodec.Frame`, and 1-D :class:`torch.Tensor` for
# :class:`~torchcodec.FrameBatch` (one value per frame in the batch).

# %%
# Using time-based indexing
# -------------------------
#
# So far, we have retrieved frames based on their index. We can also retrieve
# frames based on *when* they are played with
# :meth:`~torchcodec.decoders.VideoDecoder.get_frame_played_at` and
# :meth:`~torchcodec.decoders.VideoDecoder.get_frames_played_at`, which
# also returns :class:`~torchcodec.Frame` and :class:`~torchcodec.FrameBatch`
# respectively.

frame_at_2_seconds = decoder.get_frame_played_at(seconds=2)
print(f"{type(frame_at_2_seconds) = }")
print(frame_at_2_seconds)

# %%
other_frames = decoder.get_frames_played_at(seconds=[10.1, 0.3, 5])
print(f"{type(other_frames) = }")
print(other_frames)

# %%
plot(frame_at_2_seconds.data, "Frame played at 2 seconds")
plot(other_frames.data, "Other frames")
