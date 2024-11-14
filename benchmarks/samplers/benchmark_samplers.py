import argparse
from pathlib import Path
from time import perf_counter_ns

import torch
from torchcodec.decoders import VideoDecoder
from torchcodec.samplers import (
    clips_at_random_indices,
    clips_at_random_timestamps,
    clips_at_regular_indices,
    clips_at_regular_timestamps,
)

DEFAULT_VIDEO_PATH = Path(__file__).parent / "../../test/resources/nasa_13013.mp4"
DEFAULT_NUM_EXP = 30


def bench(f, *args, num_exp, warmup=0, seed, **kwargs):

    for _ in range(warmup):
        f(*args, **kwargs)

    num_frames = None
    times = []
    for _ in range(num_exp):
        if seed is not None:
            torch.manual_seed(seed)
        start = perf_counter_ns()
        clips = f(*args, **kwargs)
        end = perf_counter_ns()
        times.append(end - start)
        num_frames = (
            clips.data.shape[0] * clips.data.shape[1]
        )  # should be constant across calls
    return torch.tensor(times).float(), num_frames


def report_stats(times, num_frames, unit="ms"):
    fps = num_frames * 1e9 / torch.median(times)

    mul = {
        "ns": 1,
        "µs": 1e-3,
        "ms": 1e-6,
        "s": 1e-9,
    }[unit]
    times = times * mul
    std = times.std().item()
    med = times.median().item()
    print(f"{med = :.2f}{unit} +- {std:.2f}  med fps = {fps:.1f}")
    return med, fps


def sample(decoder, sampler, **kwargs):
    return sampler(
        decoder,
        num_frames_per_clip=10,
        **kwargs,
    )


def run_sampler_benchmarks(device, video, num_experiments, torch_seed):

    for num_clips in (1, 50):
        print("-" * 10)
        print(f"{num_clips = }")

        print("clips_at_random_indices     ", end="")
        decoder = VideoDecoder(video, device=device)
        times, num_frames = bench(
            sample,
            decoder,
            clips_at_random_indices,
            num_clips=num_clips,
            num_exp=num_experiments,
            warmup=2,
            seed=torch_seed,
        )
        report_stats(times, num_frames, unit="ms")

        print("clips_at_regular_indices    ", end="")
        times, num_frames = bench(
            sample,
            decoder,
            clips_at_regular_indices,
            num_clips=num_clips,
            num_exp=num_experiments,
            warmup=2,
            seed=torch_seed,
        )
        report_stats(times, num_frames, unit="ms")

        print("clips_at_random_timestamps  ", end="")
        times, num_frames = bench(
            sample,
            decoder,
            clips_at_random_timestamps,
            num_clips=num_clips,
            num_exp=num_experiments,
            warmup=2,
            seed=torch_seed,
        )
        report_stats(times, num_frames, unit="ms")

        print("clips_at_regular_timestamps ", end="")
        seconds_between_clip_starts = 13 / num_clips  # approximate. video is 13s long
        times, num_frames = bench(
            sample,
            decoder,
            clips_at_regular_timestamps,
            seconds_between_clip_starts=seconds_between_clip_starts,
            num_exp=num_experiments,
            warmup=2,
            seed=torch_seed,
        )
        report_stats(times, num_frames, unit="ms")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--video", type=str, default=str(DEFAULT_VIDEO_PATH))
    parser.add_argument("--num_experiments", type=int, default=DEFAULT_NUM_EXP)
    parser.add_argument("--torch_seed", type=int)
    args = parser.parse_args()
    run_sampler_benchmarks(
        args.device, args.video, args.num_experiments, args.torch_seed
    )


if __name__ == "__main__":
    main()
