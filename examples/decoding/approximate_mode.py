# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
===================================================================
Exact vs Approximate seek mode: Performance and accuracy comparison
===================================================================

In this example, we will describe the ``seek_mode`` parameter of the
:class:`~torchcodec.decoders.VideoDecoder` class.
This parameter offers a trade-off between the speed of the
:class:`~torchcodec.decoders.VideoDecoder` creation, against the seeking
accuracy of the retreived frames (i.e. in approximate mode, requesting the
``i``'th frame may not necessarily return frame ``i``).
"""

# %%
# First, a bit of boilerplate: we'll download a short video from the web, and
# use the ffmpeg CLI to repeat it 100 times. We'll end up with two videos: a
# short video of approximately 13s and a long one of about 20 mins.
# You can ignore that part and jump right below to :ref:`perf_creation`.


import torch
import requests
import tempfile
from pathlib import Path
import shutil
import subprocess
from time import perf_counter_ns


# Video source: https://www.pexels.com/video/dog-eating-854132/
# License: CC0. Author: Coverr.
url = "https://videos.pexels.com/video-files/854132/854132-sd_640_360_25fps.mp4"
response = requests.get(url, headers={"User-Agent": ""})
if response.status_code != 200:
    raise RuntimeError(f"Failed to download video. {response.status_code = }.")

temp_dir = tempfile.mkdtemp()
short_video_path = Path(temp_dir) / "short_video.mp4"
with open(short_video_path, 'wb') as f:
    for chunk in response.iter_content():
        f.write(chunk)

long_video_path = Path(temp_dir) / "long_video.mp4"
ffmpeg_command = [
    "ffmpeg",
    "-stream_loop", "99",  # repeat video 100 times
    "-i", f"{short_video_path}",
    "-c", "copy",
    f"{long_video_path}"
]
subprocess.run(ffmpeg_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

from torchcodec.decoders import VideoDecoder
print(f"Short video duration: {VideoDecoder(short_video_path).metadata.duration_seconds} seconds")
print(f"Long video duration: {VideoDecoder(long_video_path).metadata.duration_seconds / 60} minutes")

# %%
# .. _perf_creation:
#
# Performance: ``VideoDecoder`` creation
# --------------------------------------
#
# In terms of performance, the ``seek_mode`` parameter ultimately affects the
# **creation** of a :class:`~torchcodec.decoders.VideoDecoder` object. The
# longer the video, the higher the performance gain.


def bench(f, average_over=50, warmup=2, **f_kwargs):

    for _ in range(warmup):
        f(**f_kwargs)

    times = []
    for _ in range(average_over):
        start = perf_counter_ns()
        f(**f_kwargs)
        end = perf_counter_ns()
        times.append(end - start)

    times = torch.tensor(times) * 1e-6  # ns to ms
    std = times.std().item()
    med = times.median().item()
    print(f"{med = :.2f}ms +- {std:.2f}")


print("Creating a VideoDecoder object with seek_mode='exact' on a short video:")
bench(VideoDecoder, source=short_video_path, seek_mode="exact")
print("Creating a VideoDecoder object with seek_mode='approximate' on a short video:")
bench(VideoDecoder, source=short_video_path, seek_mode="approximate")
print()
print("Creating a VideoDecoder object with seek_mode='exact' on a long video:")
bench(VideoDecoder, source=long_video_path, seek_mode="exact")
print("Creating a VideoDecoder object with seek_mode='approximate' on a long video:")
bench(VideoDecoder, source=long_video_path, seek_mode="approximate")

# %%
# Performance: frame decoding and clip sampling
# ---------------------------------------------
#
# Strictly speaking the ``seek_mode`` parameter only affects the performance of
# the :class:`~torchcodec.decoders.VideoDecoder` creation. It does not have a
# direct effect on the performance of frame decoding or sampling.  **However**,
# because frame decoding and sampling patterns typically involve the creation of
# the :class:`~torchcodec.decoders.VideoDecoder` (one per video), ``seek_mode``
# may very well end up affecting the performance of decoding and samplers. For
# example:

from torchcodec import samplers


def sample_clips(seek_mode):
    return samplers.clips_at_random_indices(
        decoder=VideoDecoder(
            source=long_video_path,
            seek_mode=seek_mode
        ),
        num_clips=5,
        num_frames_per_clip=2,
    )


print("Sampling clips with seek_mode='exact':")
bench(sample_clips, seek_mode="exact")
print("Sampling clips with seek_mode='approximate':")
bench(sample_clips, seek_mode="approximate")

# %%
# Accuracy: Metadata and frame retrieval
# --------------------------------------
#
# We've seen that using ``seek_mode="approximate"`` can significantly speed up
# the :class:`~torchcodec.decoders.VideoDecoder` creation. The price to pay for
# that is that seeking won't always be as accurate as with
# ``seek_mode="exact"``. It can also affect the exactness of the metadata.
#
# However, in a lot of cases, you'll find that there will be no accuracy
# difference between the two modes, which means that ``seek_mode="approximate"``
# is a net win:

print("Metadata of short video with seek_mode='exact':")
print(VideoDecoder(short_video_path, seek_mode="exact").metadata)
print("Metadata of short video with seek_mode='approximate':")
print(VideoDecoder(short_video_path, seek_mode="approximate").metadata)

exact_decoder = VideoDecoder(short_video_path, seek_mode="exact")
approx_decoder = VideoDecoder(short_video_path, seek_mode="approximate")
for i in range(len(exact_decoder)):
    torch.testing.assert_close(
        exact_decoder.get_frame_at(i).data,
        approx_decoder.get_frame_at(i).data,
        atol=0, rtol=0,
    )
print("Frame seeking is the same for this video!")

# %%
# What is this doing under the hood?
# ----------------------------------
#
# With ``seek_mode="exact"``, the :class:`~torchcodec.decoders.VideoDecoder`
# performs a :term:`scan` when it is instantiated. The scan doesn't involve
# decoding, but processes an entire file to infer more accurate metadata (like
# duration), and also builds an internal index of frames and key-frames. This
# internal index is potentially more accurate than the one in the file's
# headers, which leads to more accurate seeking behavior.
# Without the scan, TorchCodec relies only on the metadata contained in the
# file, which may not always be as accurate.
#
# Which mode should I use?
# ------------------------
#
# The general rule of thumb is as follows:
#
# - If you really care about exactness of frame seeking, use "exact".
# - If you can sacrifice exactness of seeking for speed, which is usually the
#   case when doing clip sampling, use "approximate".
# - If your videos don't have variable framerate and their metadata is correct,
#   then "approximate" mode is a net win: it will be just as accurate as the
#   "exact" mode while still being significantly faster.

# %%
shutil.rmtree(temp_dir)
# %%
