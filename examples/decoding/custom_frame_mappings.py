# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
========================================================================
Decoding with custom_frame_mappings: Performance and accuracy comparison
========================================================================

In this example, we will describe the ``custom_frame_mappings`` parameter of the
:class:`~torchcodec.decoders.VideoDecoder` class.
"""

# %%
# Create an HD video using ffmpeg and use the ffmpeg CLI to repeat it 10 times
# to get two videos: a short video of approximately 30 seconds and a long one of about 10 mins.

import tempfile
from pathlib import Path
import subprocess
from torchcodec.decoders import VideoDecoder

temp_dir = tempfile.mkdtemp()
short_video_path = Path(temp_dir) / "short_video.mp4"

ffmpeg_generate_video_command = [
    "ffmpeg",
    "-y",
    "-f", "lavfi",
    "-i", "mandelbrot=s=1920x1080",
    "-t", "30",
    "-c:v", "h264",
    "-r", "60",
    "-g", "600",
    "-pix_fmt", "yuv420p",
    f"{short_video_path}"
]
subprocess.run(ffmpeg_generate_video_command)

long_video_path = Path(temp_dir) / "long_video.mp4"
ffmpeg_command = [
    "ffmpeg",
    "-stream_loop", "20",  # repeat video 20 times to get a 10 min video
    "-i", f"{short_video_path}",
    "-c", "copy",
    f"{long_video_path}"
]
subprocess.run(ffmpeg_command)

print(f"Short video duration: {VideoDecoder(short_video_path).metadata.duration_seconds} seconds")
print(f"Long video duration: {VideoDecoder(long_video_path).metadata.duration_seconds / 60} minutes")

# %%
# Preprocessing step to create frame mappings for the videos using ffprobe.

from pathlib import Path
import subprocess
import tempfile
from time import perf_counter_ns

stream_index = 0

temp_dir = tempfile.mkdtemp()
long_json_path = Path(temp_dir) / "long_custom_frame_mappings.json"
short_json_path = Path(temp_dir) / "short_custom_frame_mappings.json"

ffprobe_cmd = ["ffprobe", "-i", f"{long_video_path}", "-select_streams", f"{stream_index}", "-show_frames", "-show_entries", "frame=pts,duration,key_frame", "-of", "json"]
ffprobe_result = subprocess.run(ffprobe_cmd, check=True, capture_output=True, text=True)
with open(long_json_path, "w") as f:
    f.write(ffprobe_result.stdout)
    print(f"Wrote {len(ffprobe_result.stdout)} characters to {long_json_path}")

ffprobe_cmd = ["ffprobe", "-i", f"{short_video_path}", "-select_streams", f"{stream_index}", "-show_frames", "-show_entries", "frame=pts,duration,key_frame", "-of", "json"]
ffprobe_result = subprocess.run(ffprobe_cmd, check=True, capture_output=True, text=True)
with open(short_json_path, "w") as f:
    f.write(ffprobe_result.stdout)
    print(f"Wrote {len(ffprobe_result.stdout)} characters to {short_json_path}")

# %%
# Define benchmarking function. When a file_like object is passed in, its necessary to seek
# to the beginning of the file before reading it in the next iteration.

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
# Compare performance of initializing VideoDecoder with custom_frame_mappings vs exact seek_mode


for video_path, json_path in ((short_video_path, short_json_path), (long_video_path, long_json_path)):
    print(f"Running benchmarks on {Path(video_path).name}")
    print("Creating a VideoDecoder object with custom_frame_mappings JSON str from file:")
    with open(json_path, "r") as f:
        bench(VideoDecoder, source=video_path, stream_index=stream_index, custom_frame_mappings=(f.read()))

    print("Creating a VideoDecoder object with custom_frame_mappings from filelike:")
    with open(json_path, "r") as f:
        bench(VideoDecoder, file_like=True, source=video_path, stream_index=stream_index, custom_frame_mappings=f)

    # Compare against seek_modes
    print("Creating a VideoDecoder object with seek_mode='exact':")
    bench(VideoDecoder, source=video_path, stream_index=stream_index, seek_mode="exact")

# %%
# Decode frames with custom_frame_mappings vs exact seek_mode


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
        bench(decode_frames, video_path=video_path, custom_frame_mappings=(f.read()))

    print("Decoding frames with seek_mode='exact':")
    bench(decode_frames, video_path=video_path, seek_mode="exact")

# %%
# Compare frame accuracy with custom_frame_mappings vs exact seek_mode
video_path = short_video_path
json_path = short_json_path
with open(json_path, "r") as f:
    custom_frame_mappings = f.read()
    custom_frame_mappings_decoder = VideoDecoder(
        source=video_path,
        custom_frame_mappings=custom_frame_mappings
    )

exact_decoder = VideoDecoder(short_video_path, seek_mode="exact")
approx_decoder = VideoDecoder(short_video_path, seek_mode="approximate")

print("Metadata of short video with custom_frame_mappings:")
print(custom_frame_mappings_decoder.metadata)
print("Metadata of short video with seek_mode='exact':")
print(exact_decoder.metadata)
print("Metadata of short video with seek_mode='approximate':")
print(approx_decoder.metadata)

for i in range(len(approx_decoder)):
    torch.testing.assert_close(
        approx_decoder.get_frame_at(i).data,
        custom_frame_mappings_decoder.get_frame_at(i).data,
        atol=0, rtol=0,
    )
print("Frame seeking is the same for this video!")

# %%
