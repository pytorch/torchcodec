# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
====================================
Decoding with custom frame mappings
====================================

In this example, we will describe the ``custom_frame_mappings`` parameter of the
:class:`~torchcodec.decoders.VideoDecoder` class.
This parameter allows you to provide pre-computed frame mapping information to
speed up :class:`~torchcodec.decoders.VideoDecoder` instantiation, while
maintaining the frame seeking accuracy of ``seek_mode="exact"``.

This makes it ideal for workflows where:

    1. Frame accuracy is critical, so :doc:`approximate mode <approximate_mode>` cannot be used
    2. Videos can be preprocessed once and then decoded many times
"""

# %%
# First, some boilerplate: we'll download a short video from the web, and
# use ffmpeg to create a longer version by repeating it multiple times. We'll end up
# with two videos: a short one of approximately 3 minutes and a long one of about 13 minutes.
# You can ignore that part and jump right below to :ref:`frame_mappings_creation`.

import tempfile
from pathlib import Path
import subprocess
import requests

url = "https://download.pytorch.org/torchaudio/tutorial-assets/stream-api/NASAs_Most_Scientifically_Complex_Space_Observatory_Requires_Precision-MP4.mp4"
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
    "-stream_loop", "3",  # repeat video 3 times to get a ~13 min video
    "-i", f"{short_video_path}",
    "-c", "copy",
    f"{long_video_path}"
]
subprocess.run(ffmpeg_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

from torchcodec.decoders import VideoDecoder
print(f"Short video duration: {VideoDecoder(short_video_path).metadata.duration_seconds} seconds")
print(f"Long video duration: {VideoDecoder(long_video_path).metadata.duration_seconds / 60} minutes")

# %%
# .. _frame_mappings_creation:
#
# Creating custom frame mappings with ffprobe
# -------------------------------------------
#
# The key to using custom frame mappings is preprocessing your videos to extract
# frame timing information and keyframe indicators. We use ffprobe to generate
# JSON files containing this metadata.

from pathlib import Path
import subprocess
import tempfile
from time import perf_counter_ns

stream_index = 0

long_json_path = Path(temp_dir) / "long_custom_frame_mappings.json"
short_json_path = Path(temp_dir) / "short_custom_frame_mappings.json"

ffprobe_cmd = ["ffprobe", "-i", f"{long_video_path}", "-select_streams", f"{stream_index}", "-show_frames", "-show_entries", "frame=pkt_pts,pkt_duration,key_frame", "-of", "json"]
ffprobe_result = subprocess.run(ffprobe_cmd, check=True, capture_output=True, text=True)
with open(long_json_path, "w") as f:
    f.write(ffprobe_result.stdout)
    print(f"Wrote {len(ffprobe_result.stdout)} characters to {long_json_path}")

ffprobe_cmd = ["ffprobe", "-i", f"{short_video_path}", "-select_streams", f"{stream_index}", "-show_frames", "-show_entries", "frame=pkt_pts,pkt_duration,key_frame", "-of", "json"]
ffprobe_result = subprocess.run(ffprobe_cmd, check=True, capture_output=True, text=True)
with open(short_json_path, "w") as f:
    f.write(ffprobe_result.stdout)
    print(f"Wrote {len(ffprobe_result.stdout)} characters to {short_json_path}")

# %%
# .. _perf_creation:
#
# Performance: ``VideoDecoder`` creation
# --------------------------------------
#
# In terms of performance, custom frame mappings ultimately affect the
# **creation** of a :class:`~torchcodec.decoders.VideoDecoder` object. The
# longer the video, the higher the performance gain.
# Let's define a benchmarking function to measure performance.
# Note that when using file-like objects for custom_frame_mappings, we need to
# seek back to the beginning between iterations since the JSON data is consumed
# during VideoDecoder creation.

import torch


def bench(f, file_like=False, average_over=50, warmup=2, **f_kwargs):
    for _ in range(warmup):
        f(**f_kwargs)
        if file_like:
            f_kwargs["custom_frame_mappings"].seek(0)

    times = []
    for _ in range(average_over):
        start = perf_counter_ns()
        f(**f_kwargs)
        end = perf_counter_ns()
        times.append(end - start)
        if file_like:
            f_kwargs["custom_frame_mappings"].seek(0)

    times = torch.tensor(times) * 1e-6  # ns to ms
    std = times.std().item()
    med = times.median().item()
    print(f"{med = :.2f}ms +- {std:.2f}")

# %%
# Now let's compare the performance of creating VideoDecoder objects with custom
# frame mappings versus the exact seek mode. You'll see that custom
# frame mappings provide significant speedups, especially for longer videos.


for video_path, json_path in ((short_video_path, short_json_path), (long_video_path, long_json_path)):
    print(f"Running benchmarks on {Path(video_path).name}")

    print("Creating a VideoDecoder object with custom_frame_mappings:")
    with open(json_path, "r") as f:
        bench(VideoDecoder, file_like=True, source=video_path, stream_index=stream_index, custom_frame_mappings=f)

    # Compare against seek_modes
    print("Creating a VideoDecoder object with seek_mode='exact':")
    bench(VideoDecoder, source=video_path, stream_index=stream_index, seek_mode="exact")

# %%
# Performance: Frame decoding with custom frame mappings
# ------------------------------------------------------
#
# Although the custom_frame_mappings parameter only affects the performance of
# the :class:`~torchcodec.decoders.VideoDecoder` creation, decoding workflows
# typically involve creating a :class:`~torchcodec.decoders.VideoDecoder` instance.
# As a result, the performance benefits of custom_frame_mappings can be seen.


def decode_frames(video_path, seek_mode = "exact", custom_frame_mappings = None):
    decoder = VideoDecoder(
        source=video_path,
        seek_mode=seek_mode,
        custom_frame_mappings=custom_frame_mappings
    )
    decoder.get_frames_in_range(start=0, stop=10)


for video_path, json_path in ((short_video_path, short_json_path), (long_video_path, long_json_path)):
    print(f"Running benchmarks on {Path(video_path).name}")
    print("Decoding frames with custom_frame_mappings JSON str from file:")
    with open(json_path, "r") as f:
        bench(decode_frames, file_like=True, video_path=video_path, custom_frame_mappings=f)

    print("Decoding frames with seek_mode='exact':")
    bench(decode_frames, video_path=video_path, seek_mode="exact")

# %%
# Accuracy: Metadata and frame retrieval
# --------------------------------------
#
# We've seen that using custom frame mappings can significantly speed up
# the :class:`~torchcodec.decoders.VideoDecoder` creation. The advantage is that
# seeking is still as accurate as with ``seek_mode="exact"``.

print("Metadata of short video with custom_frame_mappings:")
with open(short_json_path, "r") as f:
    print(VideoDecoder(short_video_path, custom_frame_mappings=f).metadata)
print("Metadata of short video with seek_mode='exact':")
print(VideoDecoder(short_video_path, seek_mode="exact").metadata)

with open(short_json_path, "r") as f:
    custom_frame_mappings_decoder = VideoDecoder(short_video_path, custom_frame_mappings=f)
exact_decoder = VideoDecoder(short_video_path, seek_mode="exact")
for i in range(len(exact_decoder)):
    torch.testing.assert_close(
        exact_decoder.get_frame_at(i).data,
        custom_frame_mappings_decoder.get_frame_at(i).data,
        atol=0, rtol=0,
    )
print("Frame seeking is the same for this video!")

# %%
# How do custom_frame_mappings help?
# ----------------------------------
#
# Custom frame mappings contain the same frame index information
# that would normally be computed during the :term:`scan` operation in exact mode.
# (frame presentation timestamps (PTS), durations, and keyframe indicators)
# By providing this information to the :class:`~torchcodec.decoders.VideoDecoder`
# as a JSON, it eliminates the need for the expensive scan while preserving all the
# accuracy benefits.
#
# Which mode should I use?
# ------------------------
#
# - For fastest decoding when speed is more important than exact seeking accuracy,
# "approximate" mode is recommended.
#
# - For exact frame seeking, custom frame mappings will benefit workflows where the
#   same videos are decoded repeatedly, and some preprocessing work can be done.
#
# - For exact frame seeking without preprocessing, use "exact" mode.
#

# %%
