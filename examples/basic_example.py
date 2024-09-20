# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
========================================
Decoding a video with SimpleVideoDecoder
========================================

In this example, we'll learn how to decode a video using the
:class:`~torchcodec.decoders.SimpleVideoDecoder` class.
"""

# %%
# First, a bit of boilerplate: we'll define the location of the video we're
# going to use and a plotting utility. You can ignore that part and jump right
# below to :ref:`creating_decoder`.

from typing import Optional
import torch


# Video source: https://www.pexels.com/video/dog-eating-854132/
# License: CC0. Author: Coverr.
#
# You can download your own copy from:
#     https://videos.pexels.com/video-files/854132/854132-sd_640_360_25fps.mp4.
# We're going to load a local copy that we already downloaded.

dog_video_path = "dog_eating_854132.mp4"


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
# We can now create a decoder from the locally available video file. You can also
# dynamically download a video and provide the raw, encoded bytes as input as well.
from torchcodec.decoders import SimpleVideoDecoder

decoder = SimpleVideoDecoder(dog_video_path)

# %%
# The has not yet been decoded by the decoder, but we already have access to
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
# :class:`~torchcodec.decoders.SimpleVideoDecoder`. Frames are always of
# ``torch.uint8`` dtype.
#

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
# :meth:`~torchcodec.decoders.SimpleVideoDecoder.get_frame_at` and
# :meth:`~torchcodec.decoders.SimpleVideoDecoder.get_frames_at`  methods, which
# will return a :class:`~torchcodec.decoders.Frame` and
# :class:`~torchcodec.decoders.FrameBatch` objects respectively.

last_frame = decoder.get_frame_at(len(decoder) - 1)
print(f"{type(last_frame) = }")
print(last_frame)

# %%
middle_frames = decoder.get_frames_at(start=10, stop=20, step=2)
print(f"{type(middle_frames) = }")
print(middle_frames)

# %%
plot(last_frame.data, "Last frame")
plot(middle_frames.data, "Middle frames")

# %%
# Both :class:`~torchcodec.decoders.Frame` and
# :class:`~torchcodec.decoders.FrameBatch` have a ``data`` field, which contains
# the decoded tensor data. They also have the ``pts_seconds`` and
# ``duration_seconds`` fields which are single ints for
# :class:`~torchcodec.decoders.Frame`, and 1-D :class:`torch.Tensor` for
# :class:`~torchcodec.decoders.FrameBatch` (one value per frame in the batch).

# %%
# Using time-based indexing
# -------------------------
#
# So far, we have retrieved frames based on their index. We can also retrieve
# frames based on *when* they are displayed with
# :meth:`~torchcodec.decoders.SimpleVideoDecoder.get_frame_displayed_at` and
# :meth:`~torchcodec.decoders.SimpleVideoDecoder.get_frames_displayed_at`, which
# also returns :class:`~torchcodec.decoders.Frame` and :class:`~torchcodec.decoders.FrameBatch`
# respectively.

frame_at_2_seconds = decoder.get_frame_displayed_at(seconds=2)
print(f"{type(frame_at_2_seconds) = }")
print(frame_at_2_seconds)

# %%
first_two_seconds = decoder.get_frames_displayed_at(
    start_seconds=0,
    stop_seconds=2,
)
print(f"{type(first_two_seconds) = }")
print(first_two_seconds)

# %%
plot(frame_at_2_seconds.data, "Frame displayed at 2 seconds")
plot(first_two_seconds.data, "Frames displayed during [0, 2) seconds")
