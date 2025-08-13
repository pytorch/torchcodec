#!/usr/bin/env python3
"""
Simple multi-threaded NVDEC decoder cache benchmark.
"""

import argparse
from pathlib import Path
import torch
from time import perf_counter_ns
from torchcodec.decoders import VideoDecoder
from joblib import Parallel, delayed


def bench(f, *args, num_exp=100, warmup=0, **kwargs):
    for _ in range(warmup):
        f(*args, **kwargs)

    times = []
    for _ in range(num_exp):
        start = perf_counter_ns()
        f(*args, **kwargs)
        end = perf_counter_ns()
        times.append(end - start)
    return torch.tensor(times).float()


def report_stats(times, unit="ms"):
    mul = {
        "ns": 1,
        "µs": 1e-3,
        "ms": 1e-6,
        "s": 1e-9,
    }[unit]
    times = times * mul
    std = times.std().item()
    med = times.median().item()
    print(f"{med = :.2f}{unit} +- {std:.2f}")
    return med


def decode_videos(video_folder, num_threads, num_frames=10):
    """Decode frames from all .h264 files using multiple threads."""
    video_files = list(Path(video_folder).glob("*.h264"))
    
    def decode_single_video(video_path):
        decoder = VideoDecoder(str(video_path), device=torch.device("cuda:0"), device_variant="custom_nvdec")
        for i in range(min(num_frames, len(decoder))):
            frame = decoder.get_frame_at(i)
        return video_path.name
    
    # Use joblib to run in parallel
    Parallel(n_jobs=num_threads, backend="threading")(
        delayed(decode_single_video)(video_path) for video_path in video_files
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_folder", help="Folder with .h264 files")
    parser.add_argument("--num-threads", type=int, default=4, help="Number of threads")
    parser.add_argument("--num-frames", type=int, default=10, help="Frames per video")
    
    args = parser.parse_args()
    
    video_files = list(Path(args.video_folder).glob("*.h264"))
    print(f"Found {len(video_files)} .h264 files")
    print(f"Using {args.num_threads} threads, {args.num_frames} frames per video")
    print("Benchmarking...")
    
    times = bench(decode_videos, args.video_folder, args.num_threads, args.num_frames, warmup=0, num_exp=10)
    report_stats(times, unit="ms")


if __name__ == "__main__":
    main()