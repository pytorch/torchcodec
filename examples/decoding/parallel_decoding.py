# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
=============================================================
Parallel video decoding: multi-processing and multi-threading
=============================================================

In this tutorial, we'll explore different approaches to parallelize video
decoding of a large number of frames from a single video. We'll compare three
parallelization strategies:

1. **FFmpeg-based parallelism**: Using FFmpeg's internal threading capabilities
2. **Joblib multiprocessing**: Distributing work across multiple processes
3. **Joblib multithreading**: Using multiple threads within a single process

We'll use `joblib <https://joblib.readthedocs.io/en/latest/>`_ for
parallelization, as it provides very convenient APIs for distributing work
across multiple processes or threads. But this is just one of many ways to
parallelize work in Python. You can absolutely use a different thread or process
pool manager.
"""

# %%
# Let's first define some utility functions for benchmarking and data
# processing.  We'll also download a video and create a longer version by
# repeating it multiple times. This simulates working with long videos that
# require efficient processing. You can ignore that part and jump right below to
# :ref:`start_parallel_decoding`.

from typing import List
import torch
import requests
import tempfile
from pathlib import Path
import subprocess
from time import perf_counter_ns
from datetime import timedelta

from joblib import Parallel, delayed, cpu_count
from torchcodec.decoders import VideoDecoder


def bench(f, *args, num_exp=3, warmup=1, **kwargs):
    """Benchmark a function by running it multiple times and measuring execution time."""
    for _ in range(warmup):
        f(*args, **kwargs)

    times = []
    for _ in range(num_exp):
        start = perf_counter_ns()
        result = f(*args, **kwargs)
        end = perf_counter_ns()
        times.append(end - start)

    return torch.tensor(times).float(), result


def report_stats(times, unit="s"):
    """Report median and standard deviation of benchmark times."""
    mul = {
        "ns": 1,
        "µs": 1e-3,
        "ms": 1e-6,
        "s": 1e-9,
    }[unit]
    times = times * mul
    std = times.std().item()
    med = times.median().item()
    print(f"median = {med:.2f}{unit} ± {std:.2f}")
    return med


def split_indices(indices: List[int], num_chunks: int) -> List[List[int]]:
    """Split a list of indices into approximately equal chunks."""
    chunk_size = len(indices) // num_chunks
    chunks = []

    for i in range(num_chunks - 1):
        chunks.append(indices[i * chunk_size:(i + 1) * chunk_size])

    # Last chunk may be slightly larger
    chunks.append(indices[(num_chunks - 1) * chunk_size:])
    return chunks


def generate_long_video(temp_dir: str):
    # Video source: https://www.pexels.com/video/dog-eating-854132/
    # License: CC0. Author: Coverr.
    url = "https://videos.pexels.com/video-files/854132/854132-sd_640_360_25fps.mp4"
    response = requests.get(url, headers={"User-Agent": ""})
    if response.status_code != 200:
        raise RuntimeError(f"Failed to download video. {response.status_code = }.")

    short_video_path = Path(temp_dir) / "short_video.mp4"
    with open(short_video_path, 'wb') as f:
        for chunk in response.iter_content():
            f.write(chunk)

    # Create a longer video by repeating the short one 50 times
    long_video_path = Path(temp_dir) / "long_video.mp4"
    ffmpeg_command = [
        "ffmpeg", "-y",
        "-stream_loop", "49",  # repeat video 50 times
        "-i", str(short_video_path),
        "-c", "copy",
        str(long_video_path)
    ]
    subprocess.run(ffmpeg_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return short_video_path, long_video_path


temp_dir = tempfile.mkdtemp()
short_video_path, long_video_path = generate_long_video(temp_dir)

decoder = VideoDecoder(long_video_path, seek_mode="approximate")
metadata = decoder.metadata

short_duration = timedelta(seconds=VideoDecoder(short_video_path).metadata.duration_seconds)
long_duration = timedelta(seconds=metadata.duration_seconds)
print(f"Original video duration: {int(short_duration.total_seconds() // 60)}m{int(short_duration.total_seconds() % 60):02d}s")
print(f"Long video duration: {int(long_duration.total_seconds() // 60)}m{int(long_duration.total_seconds() % 60):02d}s")
print(f"Video resolution: {metadata.width}x{metadata.height}")
print(f"Average FPS: {metadata.average_fps:.1f}")
print(f"Total frames: {metadata.num_frames}")


# %%
# .. _start_parallel_decoding:
#
# Frame sampling strategy
# -----------------------
#
# For this tutorial, we'll sample a frame every 2 seconds from our long video.
# This simulates a common scenario where you need to process a subset of frames
# for LLM inference.

TARGET_FPS = 2
step = max(1, round(metadata.average_fps / TARGET_FPS))
all_indices = list(range(0, metadata.num_frames, step))

print(f"Sampling 1 frame every {TARGET_FPS} seconds")
print(f"We'll skip every {step} frames")
print(f"Total frames to decode: {len(all_indices)}")


# %%
# Method 1: Sequential decoding (baseline)
# ----------------------------------------
#
# Let's start with a sequential approach as our baseline. This processes
# frames one by one without any parallelization.

def decode_sequentially(indices: List[int], video_path=long_video_path):
    """Decode frames sequentially using a single decoder instance."""
    decoder = VideoDecoder(video_path, seek_mode="approximate")
    return decoder.get_frames_at(indices)


times, result_sequential = bench(decode_sequentially, all_indices)
sequential_time = report_stats(times, unit="s")


# %%
# Method 2: FFmpeg-based parallelism
# ----------------------------------
#
# FFmpeg has built-in multithreading capabilities that can be controlled
# via the ``num_ffmpeg_threads`` parameter. This approach uses multiple
# threads within FFmpeg itself to accelerate decoding operations.

def decode_with_ffmpeg_parallelism(
    indices: List[int],
    num_threads: int,
    video_path=long_video_path
):
    """Decode frames using FFmpeg's internal threading."""
    decoder = VideoDecoder(video_path, num_ffmpeg_threads=num_threads, seek_mode="approximate")
    return decoder.get_frames_at(indices)


NUM_CPUS = cpu_count()

times, result_ffmpeg = bench(decode_with_ffmpeg_parallelism, all_indices, num_threads=NUM_CPUS)
ffmpeg_time = report_stats(times, unit="s")
speedup = sequential_time / ffmpeg_time
print(f"Speedup compared to sequential: {speedup:.2f}x with {NUM_CPUS} FFmpeg threads.")


# %%
# Method 3: multiprocessing
# -------------------------
#
# Process-based parallelism distributes work across multiple Python processes.

def decode_with_multiprocessing(
    indices: List[int],
    num_processes: int,
    video_path=long_video_path
):
    """Decode frames using multiple processes with joblib."""
    chunks = split_indices(indices, num_chunks=num_processes)

    # loky is a multi-processing backend for joblib: https://github.com/joblib/loky
    results = Parallel(n_jobs=num_processes, backend="loky", verbose=0)(
        delayed(decode_sequentially)(chunk, video_path) for chunk in chunks
    )

    return torch.cat([frame_batch.data for frame_batch in results], dim=0)


times, result_multiprocessing = bench(decode_with_multiprocessing, all_indices, num_processes=NUM_CPUS)
multiprocessing_time = report_stats(times, unit="s")
speedup = sequential_time / multiprocessing_time
print(f"Speedup compared to sequential: {speedup:.2f}x with {NUM_CPUS} processes.")


# %%
# Method 4: Joblib multithreading
# -------------------------------
#
# Thread-based parallelism uses multiple threads within a single process.
# TorchCodec releases the GIL, so this can be very effective.

def decode_with_multithreading(
    indices: List[int],
    num_threads: int,
    video_path=long_video_path
):
    """Decode frames using multiple threads with joblib."""
    chunks = split_indices(indices, num_chunks=num_threads)

    results = Parallel(n_jobs=num_threads, prefer="threads", verbose=0)(
        delayed(decode_sequentially)(chunk, video_path) for chunk in chunks
    )

    # Concatenate results from all threads
    return torch.cat([frame_batch.data for frame_batch in results], dim=0)


times, result_multithreading = bench(decode_with_multithreading, all_indices, num_threads=NUM_CPUS)
multithreading_time = report_stats(times, unit="s")
speedup = sequential_time / multithreading_time
print(f"Speedup compared to sequential: {speedup:.2f}x with {NUM_CPUS} threads.")


# %%
# Validation and correctness check
# --------------------------------
#
# Let's verify that all methods produce identical results.

torch.testing.assert_close(result_sequential.data, result_ffmpeg.data, atol=0, rtol=0)
torch.testing.assert_close(result_sequential.data, result_multiprocessing, atol=0, rtol=0)
torch.testing.assert_close(result_sequential.data, result_multithreading, atol=0, rtol=0)
print("All good!")

# %%
import shutil
shutil.rmtree(temp_dir)
