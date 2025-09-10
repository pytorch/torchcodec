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
# Create an HD video using ffmpeg and use the ffmpeg CLI to repeat it 100 times.
# To get videos: a short video of approximately 30 seconds and a long one of about 22 mins.


import tempfile
from pathlib import Path
import subprocess

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
    "-stream_loop", "20",  # repeat video 20 times to get a ~20 min video
    "-i", f"{short_video_path}",
    "-c", "copy",
    f"{long_video_path}"
]
subprocess.run(ffmpeg_command)
from torchcodec.decoders import VideoDecoder
test_decoder = VideoDecoder(short_video_path)
print(f"Short video duration: {test_decoder.metadata.duration_seconds} seconds")
print(f"Long video duration: {VideoDecoder(long_video_path).metadata.duration_seconds / 60} minutes")

# %%
# Preprocessing step
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
# Define benchmarking function

from torchcodec import samplers
from torchcodec.decoders._video_decoder import VideoDecoder
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
# Compare performance of initializing VideoDecoder with custom_frame_mappings vs seek_modes


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
# Decode entire videos with custom_frame_mappings vs seek_modes

from torchcodec.decoders._video_decoder import VideoDecoder


def decode_frames(video_path, seek_mode = "exact", custom_frame_mappings = None):
    decoder = VideoDecoder(
        source=video_path,
        seek_mode=seek_mode,
        custom_frame_mappings=custom_frame_mappings
    )
    decoder.get_frames_in_range(start=0, stop=100)


for video_path, json_path in ((short_video_path, short_json_path), (long_video_path, long_json_path)):
    print(f"Running benchmarks on {Path(video_path).name}")
    print("Decoding frames with custom_frame_mappings JSON str from file:")
    with open(json_path, "r") as f:
        bench(decode_frames, video_path=video_path, custom_frame_mappings=(f.read()))

    print("Creating a VideoDecoder object with custom_frame_mappings from filelike:")
    with open(json_path, "r") as f:
        bench(decode_frames, file_like=True, video_path=video_path, custom_frame_mappings=f)

    # Compare against seek_modes
    print("Decoding frames with seek_mode='exact':")
    bench(decode_frames, video_path=video_path, seek_mode="exact")

# %%
# Compare performance of sampling clips with custom_frame_mappings vs seek_modes


def sample_clips(video_path, seek_mode = "exact", custom_frame_mappings = None):
    return samplers.clips_at_random_indices(
        decoder=VideoDecoder(
            source=video_path,
            seek_mode=seek_mode,
            custom_frame_mappings=custom_frame_mappings
        ),
        num_clips=5,
        num_frames_per_clip=2,
    )


for video_path, json_path in ((short_video_path, short_json_path), (long_video_path, long_json_path)):
    print(f"Running benchmarks on {Path(video_path).name}")
    print("Sampling clips with custom_frame_mappings:")
    with open(json_path, "r") as f:
        mappings = f.read()
        bench(sample_clips, file_like=False, video_path=video_path, custom_frame_mappings=mappings)

    print("Sampling clips with seek_mode='exact':")
    bench(sample_clips, video_path=video_path, seek_mode="exact")

# %%
