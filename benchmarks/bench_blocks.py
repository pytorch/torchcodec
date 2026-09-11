# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import queue
import subprocess
import threading
from pathlib import Path
from time import perf_counter_ns

import psutil
import torch

from torchcodec.decoders import VideoDecoder
from torchcodec.decoders._blocks import ColorConverter, Demuxer

# Kept minimal on purpose; the filename is derived from exactly these.
_DURATION_S = 10
_HEIGHT = 720
_WIDTH = 1280
_FPS = 30
_SOURCE = "testsrc2"


def make_video() -> str:
    """Generate (once) a 720p/10s test clip in /tmp, keyed by its generation
    parameters, and return its path. Reused if it already exists."""
    key = f"{_SOURCE}_{_WIDTH}x{_HEIGHT}_{_FPS}fps_{_DURATION_S}s"
    path = Path("/tmp") / f"bench_blocks_{key}.mp4"
    if not path.exists():
        lavfi = f"{_SOURCE}=size={_WIDTH}x{_HEIGHT}:rate={_FPS}:duration={_DURATION_S}"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                lavfi,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-g",
                str(_FPS),
                str(path),
            ],
            check=True,
            capture_output=True,
        )
    return str(path)


# The three decode stages, each a generator transforming an iterator of inputs
# into an iterator of outputs. They compose directly (sequential is just
# convert(decode(demux()))); a thread boundary between any two stages is inserted
# with prefetch(). Where you insert it decides which stages overlap.


def _demux(demuxer):
    yield from demuxer


def _decode(decoder, packets):
    for packet in packets:
        yield from decoder.decode(packet)
    yield from decoder.drain()


def _convert(converter, frames):
    for frame in frames:
        yield converter.convert(frame)


def prefetch(upstream, buffer_size=8):
    # Run `upstream` (a generator chaining one or more stages) on a background
    # thread, yielding its items through a bounded queue. The queue applies
    # backpressure: the worker blocks in q.put() when the buffer is full, so it
    # only runs ~buffer_size items ahead of the consumer.
    q: queue.Queue = queue.Queue(maxsize=buffer_size)
    eof = object()
    error = []

    def worker():
        try:
            for item in upstream:
                q.put(item)
        except Exception as e:  # surface failures instead of hanging
            error.append(e)
        finally:
            q.put(eof)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    def drain():
        while (item := q.get()) is not eof:
            yield item
        thread.join()  # worker enqueued eof and is finishing; make it explicit
        if error:
            raise error[0]

    return drain()


def _consume(frames):
    for _ in frames:
        pass


def _decode_sequential(path, device="cpu"):
    demuxer = Demuxer(path)
    decoder = demuxer.streams[0].make_decoder(device=device)
    converter = ColorConverter(device=device)
    _consume(_convert(converter, _decode(decoder, _demux(demuxer))))


def _decode_prefetch_frames(path, device="cpu"):
    # [demux + decode] on one thread || [color-convert] on another.
    demuxer = Demuxer(path)
    decoder = demuxer.streams[0].make_decoder(device=device)
    converter = ColorConverter(device=device)
    frames = prefetch(_decode(decoder, _demux(demuxer)))
    _consume(_convert(converter, frames))


def _decode_prefetch_packets(path, device="cpu"):
    # [demux] on one thread || [decode + color-convert] on another.
    demuxer = Demuxer(path)
    decoder = demuxer.streams[0].make_decoder(device=device)
    converter = ColorConverter(device=device)
    packets = prefetch(_demux(demuxer))
    _consume(_convert(converter, _decode(decoder, packets)))


def _decode_prefetch_packets_and_frames(path, device="cpu"):
    # [demux] || [decode] || [color-convert], each on its own thread.
    demuxer = Demuxer(path)
    decoder = demuxer.streams[0].make_decoder(device=device)
    converter = ColorConverter(device=device)
    packets = prefetch(_demux(demuxer))
    frames = prefetch(_decode(decoder, packets))
    _consume(_convert(converter, frames))


def _decode_video_decoder(path, device="cpu"):
    # approximate seek mode to match the blocks
    VideoDecoder(path, seek_mode="approximate", device=device).get_all_frames()


def get_num_frames(path):
    return VideoDecoder(path).metadata.num_frames


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------


def bench(f, *args, num_exp=10, warmup=2, **kwargs):
    process = psutil.Process()
    cuda = kwargs.get("device") == "cuda"
    for _ in range(warmup):
        f(*args, **kwargs)
    times = []
    cpu_utils = []
    for _ in range(num_exp):
        process.cpu_percent(interval=None)  # reset the measurement window
        start = perf_counter_ns()
        f(*args, **kwargs)
        if cuda:
            torch.cuda.synchronize()
        end = perf_counter_ns()
        cpu_utils.append(process.cpu_percent(interval=None))  # since reset
        times.append(end - start)
    return torch.tensor(times).float(), torch.tensor(cpu_utils).float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    device = parser.parse_args().device

    path = make_video()
    print(f"Video: {path}  ({_SOURCE} {_WIDTH}x{_HEIGHT} {_FPS}fps {_DURATION_S}s)")
    print(f"Device: {device}\n")

    methods = {
        "VideoDecoder": _decode_video_decoder,
        "sequential": _decode_sequential,
        "demux+decode || cc": _decode_prefetch_frames,
        "demux || decode+cc": _decode_prefetch_packets,
        "demux || decode || cc": _decode_prefetch_packets_and_frames,
    }

    results = {}
    for name, fn in methods.items():
        times_ns, cpu = bench(fn, path, device=device)
        results[name] = {
            "mean_ms": (times_ns / 1e6).mean().item(),
            "std_ms": (times_ns / 1e6).std().item(),
            "cpu": cpu.mean().item(),
        }

    baseline = results["VideoDecoder"]["mean_ms"]
    print(f"{'Method':<24}{'Time (ms)':>20}{'CPU %':>10}{'vs VideoDecoder':>18}")
    print("-" * 72)
    for name, r in results.items():
        time_str = f"{r['mean_ms']:.1f} +/- {r['std_ms']:.1f}"
        speedup = baseline / r["mean_ms"]
        print(f"{name:<24}{time_str:>20}{r['cpu']:>10.0f}{speedup:>17.2f}x")

    print(f"\nFrames per run: {get_num_frames(path)}")


if __name__ == "__main__":
    main()
