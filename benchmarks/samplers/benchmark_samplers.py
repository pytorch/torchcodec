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


def sample(sampler, **kwargs):
    decoder = VideoDecoder(VIDEO_PATH)
    sampler(
        decoder,
        num_frames_per_clip=10,
        **kwargs,
    )


VIDEO_PATH = Path(__file__).parent / "../../test/resources/nasa_13013.mp4"
NUM_EXP = 30

for num_clips in (1, 50):
    print("-" * 10)
    print(f"{num_clips = }")

    print("clips_at_random_indices     ", end="")
    times = bench(
        sample, clips_at_random_indices, num_clips=num_clips, num_exp=NUM_EXP, warmup=2
    )
    report_stats(times, unit="ms")

    print("clips_at_regular_indices    ", end="")
    times = bench(
        sample, clips_at_regular_indices, num_clips=num_clips, num_exp=NUM_EXP, warmup=2
    )
    report_stats(times, unit="ms")

    print("clips_at_random_timestamps  ", end="")
    times = bench(
        sample,
        clips_at_random_timestamps,
        num_clips=num_clips,
        num_exp=NUM_EXP,
        warmup=2,
    )
    report_stats(times, unit="ms")

    print("clips_at_regular_timestamps ", end="")
    seconds_between_clip_starts = 13 / num_clips  # approximate. video is 13s long
    times = bench(
        sample,
        clips_at_regular_timestamps,
        seconds_between_clip_starts=seconds_between_clip_starts,
        num_exp=NUM_EXP,
        warmup=2,
    )
    report_stats(times, unit="ms")
