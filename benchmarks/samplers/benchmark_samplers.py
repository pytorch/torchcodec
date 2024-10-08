from pathlib import Path
from time import perf_counter_ns

import torch
from torchcodec.decoders import VideoDecoder
from torchcodec.samplers import clips_at_random_indices


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


def sample(num_clips):
    decoder = VideoDecoder(VIDEO_PATH)
    clips_at_random_indices(
        decoder,
        num_clips=num_clips,
        num_frames_per_clip=10,
        num_indices_between_frames=2,
    )


VIDEO_PATH = Path(__file__).parent / "../../test/resources/nasa_13013.mp4"

times = bench(sample, num_clips=1, num_exp=30, warmup=2)
report_stats(times, unit="ms")
times = bench(sample, num_clips=50, num_exp=30, warmup=2)
report_stats(times, unit="ms")
