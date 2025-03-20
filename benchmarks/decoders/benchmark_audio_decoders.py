import subprocess

from argparse import ArgumentParser
from datetime import timedelta
from pathlib import Path
from time import perf_counter_ns

import torch
import torchaudio
from torch import Tensor
from torchaudio.io import StreamReader
from torchcodec.decoders._audio_decoder import AudioDecoder

DEFAULT_NUM_EXP = 30


def bench(f, *args, num_exp=DEFAULT_NUM_EXP, warmup=1, **kwargs) -> Tensor:

    for _ in range(warmup):
        f(*args, **kwargs)

    times = []
    for _ in range(num_exp):
        start = perf_counter_ns()
        f(*args, **kwargs)
        end = perf_counter_ns()
        times.append(end - start)
    return torch.tensor(times).float()


def report_stats(times: Tensor, unit: str = "ms", prefix: str = "") -> float:
    mul = {
        "ns": 1,
        "µs": 1e-3,
        "ms": 1e-6,
        "s": 1e-9,
    }[unit]
    times = times * mul
    std = times.std().item()
    med = times.median().item()
    mean = times.mean().item()
    min = times.min().item()
    max = times.max().item()
    print(
        f"{prefix:<40} {med = :.2f}, {mean = :.2f} +- {std:.2f}, {min = :.2f}, {max = :.2f} - in {unit}"
    )


def get_duration(path: Path) -> str:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Remove microseconds
        return str(timedelta(seconds=float(result.stdout.strip()))).split(".")[0]
    except Exception:
        return "?"


def decode_with_torchcodec(path: Path) -> None:
    AudioDecoder(path).get_samples_played_in_range(start_seconds=0, stop_seconds=None)


def decode_with_torchaudio_StreamReader(path: Path) -> None:
    reader = StreamReader(path)
    reader.add_audio_stream(frames_per_chunk=1024)
    for _ in reader.stream():
        pass


def decode_with_torchaudio_load(path: Path, backend: str) -> None:
    torchaudio.load(str(path), backend=backend)


parser = ArgumentParser()
parser.add_argument("--path", type=str, help="path to file")
parser.add_argument(
    "--num-exp",
    type=int,
    default=DEFAULT_NUM_EXP,
    help="number of runs to average over",
)

args = parser.parse_args()
path = Path(args.path)


print(
    f"Benchmarking {path.name}, duration: {get_duration(path)}, averaging over {args.num_exp} runs:"
)

times = bench(decode_with_torchcodec, path, num_exp=args.num_exp)
report_stats(times, prefix="torchcodec.AudioDecoder")

times = bench(decode_with_torchaudio_load, path, backend="ffmpeg", num_exp=args.num_exp)
report_stats(times, prefix="torchaudio.load(backend='ffmpeg')")

prefix = "torchaudio.load(backend='sox')"
try:
    times = bench(
        decode_with_torchaudio_load, path, backend="sox", num_exp=args.num_exp
    )
    report_stats(times, prefix=prefix)
except RuntimeError:
    print(f"{prefix:<40} Not supported")

times = bench(decode_with_torchaudio_StreamReader, path, num_exp=args.num_exp)
report_stats(times, prefix="torchaudio.StreamReader")
