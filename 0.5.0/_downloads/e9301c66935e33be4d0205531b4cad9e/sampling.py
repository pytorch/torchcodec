# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
=========================
How to sample video clips
=========================

In this example, we'll learn how to sample video :term:`clips` from a video. A
clip generally denotes a sequence or batch of frames, and is typically passed as
input to video models.
"""

# %%
# First, a bit of boilerplate: we'll download a video from the web, and define a
# plotting utility. You can ignore that part and jump right below to
# :ref:`sampling_tuto_start`.

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
# .. _sampling_tuto_start:
#
# Creating a decoder
# ------------------
#
# Sampling clips from a video always starts by creating a
# :class:`~torchcodec.decoders.VideoDecoder` object. If you're not already
# familiar with :class:`~torchcodec.decoders.VideoDecoder`, take a quick look
# at: :ref:`sphx_glr_generated_examples_decoding_basic_example.py`.
from torchcodec.decoders import VideoDecoder

# You can also pass a path to a local file!
decoder = VideoDecoder(raw_video_bytes)

# %%
# Sampling basics
# ---------------
#
# We can now use our decoder to sample clips. Let's first look at a simple
# example: all other samplers follow similar APIs and principles. We'll use
# :func:`~torchcodec.samplers.clips_at_random_indices` to sample clips that
# start at random indices.

from torchcodec.samplers import clips_at_random_indices

# The samplers RNG is controlled by pytorch's RNG. We set a seed for this
# tutorial to be reproducible across runs, but note that hard-coding a seed for
# a training run is generally not recommended.
torch.manual_seed(0)

clips = clips_at_random_indices(
    decoder,
    num_clips=5,
    num_frames_per_clip=4,
    num_indices_between_frames=3,
)
clips

# %%
# The output of the sampler is a sequence of clips, represented as
# :class:`~torchcodec.FrameBatch` object. In this object, we have different
# fields:
#
# - ``data``: a 5D uint8 tensor representing the frame data. Its shape is
#   (num_clips, num_frames_per_clip, ...) where ... is either (C, H, W) or (H,
#   W, C), depending on the ``dimension_order`` parameter of the
#   :class:`~torchcodec.decoders.VideoDecoder`. This is typically what would get
#   passed to the model.
# - ``pts_seconds``: a 2D float tensor of shape (num_clips, num_frames_per_clip)
#   giving the starting timestamps of each frame within each clip, in seconds.
# - ``duration_seconds``: a 2D float tensor of shape (num_clips,
#   num_frames_per_clip) giving the duration of each frame within each clip, in
#   seconds.

plot(clips[0].data)

# %%
# Indexing and manipulating clips
# -------------------------------
#
# Clips are :class:`~torchcodec.FrameBatch` objects, and they support native
# pytorch indexing semantics (including fancy indexing). This makes it easy to
# filter clips based on a given criteria. For example, from the clips above we
# can easily filter out those who start *after* a specific timestamp:
clip_starts = clips.pts_seconds[:, 0]
clip_starts

# %%
clips_starting_after_five_seconds = clips[clip_starts > 5]
clips_starting_after_five_seconds

# %%
every_other_clip = clips[::2]
every_other_clip

# %%
#
# .. note::
#   A more natural and efficient way to get clips after a given timestamp is to
#   rely on the sampling range parameters, which we'll cover later in :ref:`sampling_range`.
#
# Index-based and Time-based samplers
# -----------------------------------
#
# So far we've used  :func:`~torchcodec.samplers.clips_at_random_indices`.
# Torchcodec support additional samplers, which fall under two main categories:
#
# Index-based samplers:
#
# -  :func:`~torchcodec.samplers.clips_at_random_indices`
# -  :func:`~torchcodec.samplers.clips_at_regular_indices`
#
# Time-based samplers:
#
# -  :func:`~torchcodec.samplers.clips_at_random_timestamps`
# -  :func:`~torchcodec.samplers.clips_at_regular_timestamps`
#
# All these samplers follow similar APIs and the time-based samplers have
# analogous parameters to the index-based ones. Both samplers types generally
# offer comparable performance in terms speed.
#
# .. note::
#   Is it better to use a time-based sampler or an index-based sampler? The
#   index-based samplers have arguably slightly simpler APIs and their behavior
#   is possibly simpler to understand and control, because of the discrete
#   nature of indices. For videos with constant fps, an index-based sampler
#   behaves exactly the same as a time-based samplers. For videos with variable
#   fps however (as is often the case), relying on indices may under/over sample
#   some regions in the video, which may lead to undersirable side effects when
#   training a model. Using a time-based sampler ensures uniform sampling
#   caracteristics along the time-dimension.
#

# %%
# .. _sampling_range:
#
# Advanced parameters: sampling range
# -----------------------------------
#
# Sometimes, we may not want to sample clips from an entire video. We may only
# be interested in clips that start within a smaller interval. In samplers, the
# ``sampling_range_start`` and ``sampling_range_end`` parmeters control the
# sampling range: they define where we allow clips to *start*. There are two
# important things to keep in mind:
#
# - ``sampling_range_end`` is an *open* upper-bound: clips may only start within
#   [sampling_range_start, sampling_range_end).
# - Because these parameter define where a clip can start, clips may contain
#   frames *after*  ``sampling_range_end``!

from torchcodec.samplers import clips_at_regular_timestamps

clips = clips_at_regular_timestamps(
    decoder,
    seconds_between_clip_starts=1,
    num_frames_per_clip=4,
    seconds_between_frames=0.5,
    sampling_range_start=2,
    sampling_range_end=5
)
clips

# %%
# Advanced parameters: policy
# ---------------------------
#
# Depending on the length or duration of the video and on the sampling
# parameters, the sampler may try to sample frames *beyond* the end of the
# video. The ``policy`` parameter defines how such invalid frames should be
# replaced with valid
# frames.
from torchcodec.samplers import clips_at_random_timestamps

end_of_video = decoder.metadata.end_stream_seconds
print(f"{end_of_video = }")

# %%
torch.manual_seed(0)
clips = clips_at_random_timestamps(
    decoder,
    num_clips=1,
    num_frames_per_clip=5,
    seconds_between_frames=0.4,
    sampling_range_start=end_of_video - 1,
    sampling_range_end=end_of_video,
    policy="repeat_last",
)
clips.pts_seconds

# %%
# We see above that the end of the video is at 13.8s. The sampler tries to
# sample frames at timestamps [13.28, 13.68, 14.08, ...] but 14.08 is an invalid
# timestamp, beyond the end video. With the "repeat_last" policy, which is the
# default, the sampler simply repeats the last frame at 13.68 seconds to
# construct the clip.
#
# An alternative policy is "wrap": the sampler then wraps-around the clip and repeats the first few valid frames as necessary:

torch.manual_seed(0)
clips = clips_at_random_timestamps(
    decoder,
    num_clips=1,
    num_frames_per_clip=5,
    seconds_between_frames=0.4,
    sampling_range_start=end_of_video - 1,
    sampling_range_end=end_of_video,
    policy="wrap",
)
clips.pts_seconds

# %%
# By default, the value of ``sampling_range_end`` is automatically set such that
# the sampler *doesn't* try to sample frames beyond the end of the video: the
# default value ensures that clips start early enough before the end. This means
# that by default, the policy parameter rarely comes into action, and most users
# probably don't need to worry too much about it.
