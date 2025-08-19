#!/usr/bin/env python3

import torch
from time import perf_counter_ns
import argparse
from pathlib import Path
from torchcodec.decoders import VideoDecoder
from joblib import Parallel, delayed
import os
from contextlib import contextmanager
import torchvision.io


@contextmanager
def with_cache(enabled=True):
    """Context manager to enable/disable NVDEC decoder cache."""
    original_env_value = os.environ.get("TORCHCODEC_DISABLE_NVDEC_CACHE")
    try:
        if not enabled:
            os.environ["TORCHCODEC_DISABLE_NVDEC_CACHE"] = "1"
        elif "TORCHCODEC_DISABLE_NVDEC_CACHE" in os.environ:
            del os.environ["TORCHCODEC_DISABLE_NVDEC_CACHE"]
        yield
    finally:
        # Restore original environment variable state
        if original_env_value is not None:
            os.environ["TORCHCODEC_DISABLE_NVDEC_CACHE"] = original_env_value
        elif "TORCHCODEC_DISABLE_NVDEC_CACHE" in os.environ:
            del os.environ["TORCHCODEC_DISABLE_NVDEC_CACHE"]


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

# TODO call sync

def decode_videos_threaded(num_threads, decoder_implem):
    assert decoder_implem in ["ffmpeg", "ours"], "Invalid decoder implementation"
    device_variant = None if decoder_implem == "ffmpeg" else "custom_nvdec"
    num_frames_to_decode = 10
    
    def decode_one_video(video_path):
        device = torch.device("cuda:0")
        decoder = VideoDecoder(str(video_path), device=device, device_variant=device_variant, seek_mode="approximate")
        indices = torch.linspace(0, len(decoder)-10, num_frames_to_decode, dtype=torch.int).tolist()
        frames = decoder.get_frames_at(indices)
        return frames.data.cpu()  # Move to CPU for PNG saving
    
    # Always collect and return all decoded frames
    results = Parallel(n_jobs=num_threads, prefer="threads")(
        delayed(decode_one_video)(video_path) for video_path in video_files
    )
    torch.cuda.synchronize()
    return results


def validate_decode_correctness(video_path, num_threads=1):
    """Save decoded frames from different implementations for visual comparison."""
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Test single video with different implementations
    global video_files
    original_files = video_files
    video_files = [Path(video_path)]  # Override for single video test
    
    try:
        # Get frames from each implementation (results is a list from joblib)
        frames_ffmpeg = decode_videos_threaded(num_threads, "ffmpeg")[0]  # First (and only) video
        
        with with_cache(enabled=True):
            frames_ours_cached = decode_videos_threaded(num_threads, "ours")[0]
        
        with with_cache(enabled=False):
            frames_ours_nocache = decode_videos_threaded(num_threads, "ours")[0]
        
        # Frames are already uint8, no conversion needed
        print(f"Frame shapes: ffmpeg={frames_ffmpeg.shape}, cached={frames_ours_cached.shape}, nocache={frames_ours_nocache.shape}")
        
        # Save concatenated frames for visual comparison
        num_frames = frames_ffmpeg.shape[0]
        for i in range(min(5, num_frames)):  # Save first 5 frames
            # Frames are already [N, C, H, W], so just select frame i
            frame_ffmpeg = frames_ffmpeg[i]    # Shape: [C, H, W]
            frame_cached = frames_ours_cached[i] 
            frame_nocache = frames_ours_nocache[i]
            
            # Concatenate along width dimension (dim=2)
            concat_frame = torch.cat([frame_ffmpeg, frame_cached, frame_nocache], dim=2)
            
            output_path = results_dir / f"frame_{i:02d}_comparison.png"
            torchvision.io.write_png(concat_frame, str(output_path))
        
    finally:
        video_files = original_files  # Restore original file list


parser = argparse.ArgumentParser()
parser.add_argument("video_folder", help="Folder containing .h264 files")
parser.add_argument("--num-threads", type=int, help="Number of threads")
args = parser.parse_args()

video_files = list(Path(args.video_folder).glob("*.mp4"))
print(f"Decoder a few frames from {len(video_files)} video files in {args.video_folder} with {args.num_threads} threads")

# validate_decode_correctness(video_files[0], num_threads=args.num_threads)

print("=== Benchmarking FFmpeg backend ===")
times = bench(decode_videos_threaded, args.num_threads, decoder_implem="ffmpeg", warmup=0, num_exp=10)
report_stats(times)

print("\n=== Benchmarking our backend WITH cache ===")
with with_cache(enabled=True):
    times = bench(decode_videos_threaded, args.num_threads, decoder_implem="ours", warmup=0, num_exp=10)
    report_stats(times)

print("\n=== Benchmarking our backend WITHOUT cache ===")
with with_cache(enabled=False):
    times = bench(decode_videos_threaded, args.num_threads, decoder_implem="ours", warmup=0, num_exp=10)
    report_stats(times)