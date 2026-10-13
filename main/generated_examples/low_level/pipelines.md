# Multi-threaded decoding pipelines

Important

**The low-level APIs are in beta.** Their signatures and semantics may still
change slightly, in response to user feedback. Please [share your feedback](https://github.com/meta-pytorch/torchcodec/issues?q=is:open+is:issue)!

In this tutorial, we'll assemble the three decoding stages into pipelines of our
own: running demuxing, decoding and color-conversion concurrently on several
threads, choosing where to split them. Each of these steps individually release
the GIL.

Important

These objects can cross threads, but not processes, so multi-processing
is currently not supported. But it *can* be: if that's something you need,
please open an issue.

Note

The pipelines below run every stage on the same CUDA stream, and nothing here
requires you to think about stream synchronization. If you decide however to
give each stage a CUDA stream of its own, read
[Low-level APIs and CUDA streams synchronization](cuda_streams.html#sphx-glr-generated-examples-low-level-cuda-streams-py) for common gotchas
and how to avoid them.

Some boilerplate first: a test video, and the device we'll run on.

```
import subprocess
import tempfile
from pathlib import Path

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device = }")

temp_dir = Path(tempfile.mkdtemp())
video_path = temp_dir / "video.mp4"
subprocess.run(
 [
 "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
 "-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=30:duration=10",
 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "30",
 "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
 str(video_path),
 ],
 check=True,
)
```

```
device = 'cuda'

CompletedProcess(args=['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=1920x1080:rate=30:duration=10', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-g', '30', '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '/tmp/tmpf1qt1uip/video.mp4'], returncode=0)
```

## One stage, one generator

Each stage can be written as a generator: one over the [`Packet`](../../generated/torchcodec.decoders.Packet.html#torchcodec.decoders.Packet) objects
a [`Demuxer`](../../generated/torchcodec.decoders.Demuxer.html#torchcodec.decoders.Demuxer) produces, one over the [`RawFrame`](../../generated/torchcodec.decoders.RawFrame.html#torchcodec.decoders.RawFrame) objects a
[`VideoPacketDecoder`](../../generated/torchcodec.decoders.VideoPacketDecoder.html#torchcodec.decoders.VideoPacketDecoder) decodes them into, and one over the RGB
[`Frame`](../../generated/torchcodec.Frame.html#torchcodec.Frame) objects a [`ColorConverter`](../../generated/torchcodec.decoders.ColorConverter.html#torchcodec.decoders.ColorConverter) makes of those. A
pipeline is then a chain of generators, and inserting `prefetch()` between
two of them puts everything upstream on its own thread: the stages run
concurrently, and since the stages release the GIL, that's real parallelism.

```
import queue
import threading

from torchcodec.decoders import ColorConverter, Demuxer

def demux(demuxer):
 yield from demuxer

def decode(packet_decoder, packets):
 for packet in packets:
 yield from packet_decoder.decode(packet)
 yield from packet_decoder.drain()

def color_convert(color_converter, raw_frames):
 for raw_frame in raw_frames:
 yield color_converter.convert(raw_frame)

def prefetch(upstream, buffer_size=8):
 # Run `upstream` on a background thread, yielding its items through a
 # bounded queue.
 q = queue.Queue(maxsize=buffer_size)
 eof = object()

 def worker():
 for item in upstream:
 q.put(item)
 q.put(eof)

 threading.Thread(target=worker, daemon=True).start()

 def drain():
 while (item := q.get()) is not eof:
 yield item

 return drain()
```

## Which stage to parallelize

With those in hand, a pipeline is one expression, and moving the thread
boundary is moving one `prefetch()` call.

```
def sequential(device):
 # demux -> decode -> color-convert, all on the calling thread.
 demuxer = Demuxer(video_path)
 packet_decoder = demuxer.streams[0].make_decoder(device=device)
 color_converter = ColorConverter(device=device)
 return color_convert(color_converter, decode(packet_decoder, demux(demuxer)))

def convert_on_own_thread(device):
 # [demux + decode] on one thread || [color-convert] on another.
 demuxer = Demuxer(video_path)
 packet_decoder = demuxer.streams[0].make_decoder(device=device)
 color_converter = ColorConverter(device=device)
 raw_frames = prefetch(decode(packet_decoder, demux(demuxer)))
 return color_convert(color_converter, raw_frames)

def demux_on_own_thread(device):
 # [demux] on one thread || [decode + color-convert] on another.
 demuxer = Demuxer(video_path)
 packet_decoder = demuxer.streams[0].make_decoder(device=device)
 color_converter = ColorConverter(device=device)
 packets = prefetch(demux(demuxer))
 return color_convert(color_converter, decode(packet_decoder, packets))

def one_thread_each(device):
 # [demux] || [decode] || [color-convert], a thread per stage.
 demuxer = Demuxer(video_path)
 packet_decoder = demuxer.streams[0].make_decoder(device=device)
 color_converter = ColorConverter(device=device)
 packets = prefetch(demux(demuxer))
 raw_frames = prefetch(decode(packet_decoder, packets))
 return color_convert(color_converter, raw_frames)

PIPELINES = (sequential, convert_on_own_thread, demux_on_own_thread, one_thread_each)
for pipeline in PIPELINES:
 num_frames = 0
 for frame in pipeline(device):
 num_frames += 1
 print(f"{pipeline.__name__}: {num_frames} frames on {frame.data.device}")
```

```
sequential: 300 frames on cuda:0
convert_on_own_thread: 300 frames on cuda:0
demux_on_own_thread: 300 frames on cuda:0
one_thread_each: 300 frames on cuda:0
```

Which split is best depends on where the work is:

- On the **CPU**, color conversion costs a sizeable fraction of what decoding
costs, so `convert_on_own_thread` is usually the best one (see benchmarks
below)
- On **CUDA**, color conversion is comparatively much cheaper and is dwarfed
by the decoding time, so demuxing in parallel with `demux_on_own_thread`
may be the better split.

Let's compare the speedup that `convert_on_own_thread` gives on the CPU, vs
the sequential pipeline and a [`VideoDecoder`](../../generated/torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder) as
baselines. We iterate over the `VideoDecoder` frame by frame instead of
calling [`get_all_frames()`](../../generated/torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder.get_all_frames), so that all
three have the same memory profile: one frame at a time, rather than the
entire video. `get_all_frames()` can be faster than iterating, but it has to
hold all the frames at once.

```
from time import perf_counter_ns

from torchcodec.decoders import VideoDecoder

def bench(f, num_exp=3, warmup=1):
 for _ in range(warmup):
 f()
 times = []
 for _ in range(num_exp):
 start = perf_counter_ns()
 f()
 times.append(perf_counter_ns() - start)
 return torch.tensor(times).float().median().item() / 1e9

def consume(frames):
 for _ in frames:
 pass

def decode_all_with_videodecoder():
 decoder = VideoDecoder(video_path, device="cpu", seek_mode="approximate")
 consume(decoder)

baseline = bench(decode_all_with_videodecoder)
print(f"{'VideoDecoder':<29}: {baseline:.2f}s")

for pipeline in (sequential, convert_on_own_thread):
 seconds = bench(lambda p=pipeline: consume(p("cpu")))
 print(f"{pipeline.__name__:<29}: {seconds:.2f}s "
 f"({baseline / seconds:.2f}x vs VideoDecoder)")
```

```
VideoDecoder : 1.88s
sequential : 1.89s (0.99x vs VideoDecoder)
convert_on_own_thread : 1.62s (1.16x vs VideoDecoder)
```

`sequential` should land close to the `VideoDecoder` baseline, as
expected: they do the same work on one thread.
`convert_on_own_thread` should be faster, because it can overlap the
two most expensive steps: decoding and color-conversion. Note: actual speedup
will depend on the capabilities of the machine building these docs!

**Total running time of the script:** (0 minutes 26.622 seconds)

[`Download Jupyter notebook: pipelines.ipynb`](../../_downloads/b6cf9084a4d02eed0052adaf54947e49/pipelines.ipynb)

[`Download Python source code: pipelines.py`](../../_downloads/579a0f981c9f88f32461914ba6fac457/pipelines.py)

[`Download zipped: pipelines.zip`](../../_downloads/8bbc29b4bf5b636a2ba6868308033471/pipelines.zip)

[Gallery generated by Sphinx-Gallery](https://sphinx-gallery.github.io)