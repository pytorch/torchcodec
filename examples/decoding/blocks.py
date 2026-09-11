# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
========================================
Blocks: build your own decoding pipeline
========================================

.. warning::

   **The Blocks APIs are under active construction.** They are private
   and unreleased. Signatures and semantics may change without notice. This
   tutorial only exists to show what they will eventually make possible.

:class:`~torchcodec.decoders.VideoDecoder` is a single box that does demuxing,
decoding and color conversion for you. The Blocks APIs expose those three
stages separately:

.. code-block::

   Demuxer  ->  VideoPacketDecoder  ->  ColorConverter
    Packet         RawFrame               RGB Frame

The blocks are passive: they never create threads, and they release the GIL.
You decide how they are composed, on which threads, and where to stop. Below
we illustrate a few things this enables: overlapping stages on multiple
threads, accessing raw (YUV) frames, and decoding streams of unknown -
possibly infinite - length.

Audio works the same way, through ``AudioPacketDecoder`` and
``AudioConverter``; we come back to it at the end, and to following the audio
*and* the video of a container in a single pass over it.
"""

# %%
# Boilerplate: a test video, and the device we'll run on.
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
        "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=5",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "30",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        str(video_path),
    ],
    check=True,
)

# %%
# The three blocks
# ----------------
#
# A pipeline is just a loop. The decoder may need more than one packet before
# it can output a frame, and it buffers a few frames that ``drain()`` returns
# at the end.
#
# A ``Demuxer`` follows one or more streams of a container - by default the
# best video stream - and ``demuxer.streams`` is what it ended up with. A packet
# decoder is built from one of those streams, which is what binds it to the
# codec parameters it has to decode with.
#
# ``make_decoder()`` and ``ColorConverter`` both accept ``device="cuda"``:
# decoding then runs on NVDEC and the color conversion on the GPU, and the
# frames never leave the device. Demuxing always happens on the CPU. Left
# unspecified, ``device`` is the current default device.
from torchcodec.decoders._blocks import ColorConverter, Demuxer

demuxer = Demuxer(video_path)
(video_stream,) = demuxer.streams
packet_decoder = video_stream.make_decoder(device=device)
color_converter = ColorConverter(device=device)

frames = []
for packet in demuxer:
    for raw_frame in packet_decoder.decode(packet):
        frames.append(color_converter.convert(raw_frame))
for raw_frame in packet_decoder.drain():
    frames.append(color_converter.convert(raw_frame))

print(f"{len(frames)} frames, {frames[0].data.shape = }, "
      f"{frames[0].pts_seconds = }, {frames[0].data.device = }")

# %%
# Threading: overlapping the stages
# ---------------------------------
#
# Each stage is a generator, so a pipeline is a chain of generators. Inserting
# ``prefetch()`` between two of them puts everything upstream on its own
# thread: the stages then run concurrently, and since the blocks release the
# GIL, that's real parallelism.
import queue
import threading


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
    # bounded queue. The queue applies backpressure: the worker blocks in
    # put() when the buffer is full, so it stays at most buffer_size ahead.
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


def sequential():
    # demux -> decode -> color-convert, all on the calling thread.
    demuxer = Demuxer(video_path)
    packet_decoder = demuxer.streams[0].make_decoder(device=device)
    color_converter = ColorConverter(device=device)
    return color_convert(color_converter, decode(packet_decoder, demux(demuxer)))


def convert_on_own_thread():
    # [demux + decode] on one thread || [color-convert] on another.
    demuxer = Demuxer(video_path)
    packet_decoder = demuxer.streams[0].make_decoder(device=device)
    color_converter = ColorConverter(device=device)
    raw_frames = prefetch(decode(packet_decoder, demux(demuxer)))
    return color_convert(color_converter, raw_frames)


def demux_on_own_thread():
    # [demux] on one thread || [decode + color-convert] on another. This is the
    # natural split on CUDA: demuxing is CPU and I/O work, while decoding and
    # color conversion both happen on the GPU, so they belong together.
    demuxer = Demuxer(video_path)
    packet_decoder = demuxer.streams[0].make_decoder(device=device)
    color_converter = ColorConverter(device=device)
    packets = prefetch(demux(demuxer))
    return color_convert(color_converter, decode(packet_decoder, packets))


for pipeline in (sequential, convert_on_own_thread, demux_on_own_thread):
    frames = list(pipeline())
    print(f"{pipeline.__name__}: {len(frames)} frames on {frames[0].data.device}")

# %%
# Where you insert the thread boundaries is up to you, and so is everything
# else: nothing stops you from running one pipeline per file, decoding on the
# CPU while color-converting on the GPU, or feeding frames into your own
# pre-fetching data loader.

# %%
# Seeking
# -------
#
# ``Demuxer.seek()`` moves the demuxer to a timestamp. A decoder can only start
# on a keyframe, so the seek lands on the keyframe at or before the target, and
# the first frames that come out usually precede it: keep decoding forward and
# drop them until you reach the timestamp you asked for.
#
# The seek also invalidates the frames the decoder is holding on to, so the
# ``VideoPacketDecoder`` must be ``reset()``.
demuxer = Demuxer(video_path)
packet_decoder = demuxer.streams[0].make_decoder(device=device)
color_converter = ColorConverter(device=device)

seconds = 2.5
demuxer.seek(seconds)
packet_decoder.reset()

frames = color_convert(color_converter, decode(packet_decoder, demux(demuxer)))
landed_on = next(frames)
# The frame *playing* at a timestamp is the first one that hasn't finished
# playing by then. Not `pts_seconds >= seconds`, which skips it whenever the
# timestamp falls inside a frame rather than on its boundary.
target = next(
    frame
    for frame in frames
    if frame.pts_seconds + frame.duration_seconds > seconds
)
print(f"asked for {seconds}s, landed on {landed_on.pts_seconds:.3f}s, "
      f"target frame at {target.pts_seconds:.3f}s")

# %%
# Scanning
# --------
#
# ``VideoStream.scan()`` demuxes the whole stream once, without decoding anything,
# and returns a ``FrameIndex``: one entry per frame, in presentation order.
# This is the only way to know a stream's exact frame count, timestamps and
# keyframe positions - a container header can be wrong about all of them. It
# costs one pass over the file, and it leaves the demuxer back at the start.
#
# The ``_from_content`` suffix marks the values the header also claims to know:
# it is there so that a call site says which of the two sources it trusts.
demuxer = Demuxer(video_path)
(video_stream,) = demuxer.streams
index = video_stream.scan()
packet_decoder = video_stream.make_decoder(device=device)
color_converter = ColorConverter(device=device)

print(f"{index.num_frames_from_content} frames at "
      f"{index.average_fps_from_content} fps, "
      f"from {index.begin_stream_seconds_from_content}s "
      f"to {index.end_stream_seconds_from_content}s")

# %%
# The index is also what gives the blocks frame *indices*, which they otherwise
# don't have at all: ``index_at()`` maps a timestamp to the frame on screen
# then, and ``pts_seconds`` maps back. That's enough to build ``get_frame_at``,
# or a clip sampler, on top of the blocks.
i = index.index_at(seconds)
print(f"frame {i} is on screen at {seconds}s, and starts at {index.pts_seconds[i]}s")

# %%
# Keyframes
# ^^^^^^^^^
#
# ``is_key_frame`` is a mask over all the frames, and ``key_frame_indices`` the
# same thing as a list of indices. Keyframes are the frames that decode on
# their own, which makes them the cheap ones to reach: seeking to a keyframe's
# timestamp lands exactly on it, and the very next frame out of the decoder is
# the one you asked for - no decoding forward, nothing to drop.
#
# That makes a keyframe-only sampler about as cheap as video decoding gets,
# which is what you want for thumbnails or coarse previews.
key_frames = index.key_frame_indices
print(f"{len(key_frames)} keyframes out of {len(index)} frames, "
      f"at {index.pts_seconds[key_frames].tolist()}")

thumbnails = []
for k in key_frames.tolist():
    demuxer.seek(index.pts_seconds[k])
    packet_decoder.reset()
    raw_frame = next(decode(packet_decoder, demux(demuxer)))
    thumbnails.append(color_converter.convert(raw_frame))

print(f"{len(thumbnails)} thumbnails at "
      f"{[round(f.pts_seconds, 3) for f in thumbnails]}")

# %%
# Exact seeking
# ^^^^^^^^^^^^^
#
# One last thing the index buys you, which most files never need.
# ``seek(seconds)`` already lands on the keyframe at or before the target - but
# FFmpeg resolves a seek against *decode* timestamps, so on a file whose
# keyframes are reordered it can land on one that is *displayed after* the
# target, leaving the frames in between unreachable however far forward you
# decode. ``key_frame_seconds_for()`` gives you that keyframe's own timestamp
# instead, which always lands where you meant.
#
# The two are the same call on the overwhelming majority of files; this is the
# whole of what separates :class:`~torchcodec.decoders.VideoDecoder`'s
# ``seek_mode="exact"`` from ``"approximate"`` for a timestamp lookup.
demuxer.seek(index.key_frame_seconds_for(seconds))
packet_decoder.reset()

frames = color_convert(color_converter, decode(packet_decoder, demux(demuxer)))
target = next(f for f in frames if f.pts_seconds >= index.pts_seconds[i])
print(f"frame {i} at {target.pts_seconds:.3f}s")

# %%
# Raw frames
# ----------
#
# Color conversion is optional. A ``RawFrame`` can hand out the decoder's own
# planes as tensor views, with no copy and no conversion.
demuxer = Demuxer(video_path)
packet_decoder = demuxer.streams[0].make_decoder(device=device)
raw_frame = next(decode(packet_decoder, demux(demuxer)))

Y, U, V = raw_frame.planes
print(f"{raw_frame.pix_fmt = }, {raw_frame.bit_depth = }, "
      f"{raw_frame.colorspace = }, {raw_frame.color_range = }")
print(f"{Y.shape = }, {U.shape = }, {Y.dtype = }, {Y.stride() = }")

# %%
# These are views into the frame's memory: the row stride is the decoder's own
# line size, and the chroma planes of an NVDEC surface are two interleaved
# views over a single plane. Writing through them is visible downstream.
#
# Being the decoder's own planes, they are also never rotated - a video whose
# container asks for a rotation gives you the samples as they were encoded, and
# ``raw_frame.rotation_degrees`` tells you what to apply. ``ColorConverter``
# applies it for you.
#
# So we can do the color conversion ourselves. Here it's plain PyTorch ops -
# it could just as well be a Triton or CUDA kernel, fused with whatever your
# model needs next.
assert raw_frame.pix_fmt in ("yuv420p", "nv12")  # 8-bit 4:2:0, on CPU and CUDA
assert raw_frame.colorspace == "bt709" and raw_frame.color_range == "tv"


def yuv420_to_rgb(Y, U, V):
    # BT.709, limited range. Chroma is upsampled by nearest neighbour.
    height, width = Y.shape

    def upsample(plane):
        plane = (plane.float() - 128) * (255 / 224)
        return plane.repeat_interleave(2, 0).repeat_interleave(2, 1)[:height, :width]

    y = (Y.float() - 16) * (255 / 219)
    u, v = upsample(U), upsample(V)
    rgb = torch.stack(
        [
            y + 1.5748 * v,
            y - 0.1873 * u - 0.4681 * v,
            y + 1.8556 * u,
        ]
    )
    return rgb.round_().clamp_(0, 255).to(torch.uint8)


ours = yuv420_to_rgb(Y, U, V)
reference = ColorConverter(device=device).convert(raw_frame).data
print(f"{ours.shape = }, mean abs diff vs ColorConverter: "
      f"{(ours.float() - reference.float()).abs().mean():.2f}")

# %%
# Raw HDR frames
# --------------
#
# Raw planes come at the source's own precision, so a 10-bit HDR video gives
# ``uint16`` planes with all 10 bits intact - no clipping to 8 bits, and no
# tone mapping.
hdr_video_path = temp_dir / "hdr.mp4"
subprocess.run(
    [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=1",
        "-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-preset", "ultrafast",
        "-x265-params",
        "colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:range=limited",
        str(hdr_video_path),
    ],
    check=True,
    capture_output=True,  # x265 logs its banner to stderr no matter what
)

hdr_demuxer = Demuxer(hdr_video_path)
hdr_packet_decoder = hdr_demuxer.streams[0].make_decoder(device=device)
hdr_raw = next(decode(hdr_packet_decoder, demux(hdr_demuxer)))

hdr_Y = hdr_raw.planes[0]
print(f"{hdr_raw.pix_fmt = }, {hdr_raw.bit_depth = }, "
      f"{hdr_raw.colorspace = }, {hdr_Y.dtype = }")

# NVDEC surfaces are 16-bit containers holding the samples msb-aligned, so the
# 10 bits sit at the top and the low 6 are zero. Shift them back down to read
# the sample values.
shift = 16 - hdr_raw.bit_depth if device == "cuda" else 0
samples = hdr_Y.to(torch.int32) >> shift
print(f"luma range: [{samples.min()}, {samples.max()}], "
      f"{2 ** hdr_raw.bit_depth} levels available")

# %%
# Streams of unknown length
# -------------------------
#
# :class:`~torchcodec.decoders.VideoDecoder` needs a finite, seekable source:
# it relies on the stream's duration and frame count, and in its default
# ``seek_mode="exact"`` it scans the entire file up-front. The blocks never do
# that - they consume packets as they arrive - so they can decode a source
# that has no duration, no frame count, and no end.
#
# Let's make one: FFmpeg generating frames forever into a named pipe.
import os

fifo_path = temp_dir / "live.ts"
os.mkfifo(fifo_path)


def start_live_stream():
    return subprocess.Popen(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc2=size=640x480:rate=30",  # no duration!
            "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
            "-g", "30", "-f", "mpegts", "-y", str(fifo_path),
        ],
    )


# %%
# ``VideoDecoder`` can't do anything with that (we ask for the approximate
# seek mode; the exact one would scan the stream forever):
from torchcodec.decoders import VideoDecoder

ffmpeg = start_live_stream()
try:
    VideoDecoder(fifo_path, seek_mode="approximate")
except Exception as e:
    print(f"{type(e).__name__}: {str(e).splitlines()[0]}")
ffmpeg.kill()
ffmpeg.wait()

# %%
# The blocks just stream it, and we stop whenever we want:
ffmpeg = start_live_stream()
demuxer = Demuxer(fifo_path)
packet_decoder = demuxer.streams[0].make_decoder(device=device)
color_converter = ColorConverter(device=device)

frames = []
for frame in color_convert(color_converter, decode(packet_decoder, demux(demuxer))):
    frames.append(frame)
    if len(frames) == 100:
        break  # the stream is still going; we're the ones walking away

print(f"{len(frames)} frames, from pts {frames[0].pts_seconds:.2f}s to "
      f"{frames[-1].pts_seconds:.2f}s, {frames[0].data.shape = }")

ffmpeg.kill()
ffmpeg.wait()

# %%
# Audio
# -----
#
# Audio has the same three stages:
#
# .. code-block::
#
#    Demuxer  ->  AudioPacketDecoder  ->  AudioConverter
#     Packet        RawAudioSamples          AudioSamples
#
# Decoding is the same operation either way, so the two packet decoders are a
# single class in C++; they are separate in Python because what they hand out
# isn't. Audio comes out as ``RawAudioSamples``: the codec's own samples, in
# the codec's own sample type, as a ``[num_channels, num_samples]`` tensor.
from torchcodec.decoders._blocks import AudioConverter

audio_path = temp_dir / "audio.wav"
subprocess.run(
    [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=5",
        "-c:a", "pcm_s16le", str(audio_path),
    ],
    check=True,
)

demuxer = Demuxer(audio_path, streams="audio")
packet_decoder = demuxer.streams[0].make_decoder()
raw = next(iter(packet_decoder.decode(next(iter(demuxer)))))
print(f"{raw.sample_format = }, {raw.data.dtype = }, {raw.data.shape = }, "
      f"{raw.sample_rate = }")

# %%
# Those are the true source samples - 16-bit integers here, not floats in
# ``[-1, 1]``. ``AudioConverter`` is what normalizes them, and it can resample
# and change the channel count on the way.
#
# It differs from ``ColorConverter`` in one important way: resampling is an
# interpolation filter, so the sample it emits at a given instant depends on
# input samples on *both* sides of it. The converter therefore holds the tail
# of each frame back until the next one arrives - which is why ``convert()``
# can return fewer samples than it was given, and why the pipeline ends with
# ``drain()``. Leave that call out and you lose the end of the stream.
demuxer = Demuxer(audio_path, streams="audio")
packet_decoder = demuxer.streams[0].make_decoder()
audio_converter = AudioConverter(sample_rate=16_000, num_channels=1)

chunks = []
for packet in demuxer:
    chunks += [audio_converter.convert(raw) for raw in packet_decoder.decode(packet)]
chunks += [audio_converter.convert(raw) for raw in packet_decoder.drain()]
chunks.append(audio_converter.drain())  # don't forget me

samples = torch.cat([chunk.data for chunk in chunks], dim=1)
print(f"{samples.shape = }, {samples.dtype = }, "
      f"{chunks[-1].data.shape[1]} samples came out of drain()")

# %%
# Metadata
# --------
#
# Metadata comes in tiers, and the blocks never merge them. ``demuxer.metadata``
# is about the container, ``stream.metadata`` is what its header says about one
# stream, and ``scan()`` is what that stream's packets actually say. A name tells
# you which tier you are reading, so a header value is never mistaken for an
# exact one:
demuxer = Demuxer(video_path)
(video,) = demuxer.streams

print(f"container: {demuxer.metadata.duration_seconds_from_header}s")
print(f"header:    {video.metadata.width}x{video.metadata.height}, "
      f"{video.metadata.num_frames_from_header} frames "
      f"at {video.metadata.average_fps_from_header} fps")
print(f"content:   {video.scan().num_frames_from_content} frames "
      f"at {video.scan().average_fps_from_content} fps")

# %%
# This is deliberately less helpful than
# :attr:`~torchcodec.decoders.VideoDecoder.metadata`, which merges the two
# behind a single ``num_frames`` and picks for you. Here you pick, because
# knowing which source a number came from is the reason to be at this level.
#
# To find out what is in a file before deciding which streams to demux - a
# demuxer's streams are fixed when you construct it - use
# ``get_container_metadata()``, which reads the header and no packets. It also
# reports streams a demuxer cannot follow, such as subtitles.
from torchcodec.decoders._blocks import get_container_metadata

for stream in get_container_metadata(video_path).streams:
    print(f"  stream {stream.stream_index}: {stream.codec}")

# %%
# Video and audio together
# ------------------------
#
# One demuxer per stream means one open container per stream, so decoding both
# streams of a file that way reads it twice. A single ``Demuxer`` can follow
# several streams of one container instead, in one pass.
av_path = temp_dir / "av.mp4"
subprocess.run(
    [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path), "-i", str(audio_path),
        "-c:v", "copy", "-c:a", "aac", "-shortest", str(av_path),
    ],
    check=True,
)

# %%
# Which streams to follow is said once, at construction: a selector is
# ``"video"``, ``"audio"`` or a stream index, and ``demuxer.streams`` comes back
# in the order you asked for. Each stream builds its own decoder.
demuxer = Demuxer(av_path, streams=("video", "audio"))
video_stream, audio_stream = demuxer.streams

decoders = {
    video_stream.index: video_stream.make_decoder(device=device),
    audio_stream.index: audio_stream.make_decoder(),
}
color_converter = ColorConverter(device=device)
audio_converter = AudioConverter()

# %%
# The packets come out interleaved, in the order the container stores them, and
# ``packet.stream_index`` says which stream each belongs to - so the loop is the
# single-stream one with a dispatch in the middle. Each decoder is drained at
# the end, as always.
frames, chunks = [], []
for packet in demuxer:
    outputs = decoders[packet.stream_index].decode(packet)
    if packet.stream_index == video_stream.index:
        frames += [color_converter.convert(raw) for raw in outputs]
    else:
        chunks += [audio_converter.convert(raw) for raw in outputs]

frames += [color_converter.convert(raw)
           for raw in decoders[video_stream.index].drain()]
chunks += [audio_converter.convert(raw)
           for raw in decoders[audio_stream.index].drain()]
chunks.append(audio_converter.drain())

samples = torch.cat([chunk.data for chunk in chunks], dim=1)
print(f"{len(frames)} frames up to {frames[-1].pts_seconds:.2f}s, "
      f"{samples.shape[1]} samples in one pass over the file")

# %%
# Nothing is buffered per stream: a packet is handed to you once, and if you
# want to consume one stream at a time you hold on to the other stream's packets
# yourself. How many to keep in flight is a decision that belongs to your
# pipeline, not to the block.
#
# ``seek()`` moves the container, so every stream jumps together and every
# decoder has to be reset, not just one. ``scan()`` lives on the video streams,
# and has to be called before any packet is demuxed.

# %%
# .. warning::
#
#    These blocks do no pre-roll. A lossy codec's first frames after a seek are
#    subtly wrong until it re-primes, and a resampler started mid-stream emits
#    samples on a grid of its own, so samples decoded after a
#    ``demuxer.seek()`` do not line up bit-for-bit with the same region of a
#    whole-file decode. Decoding a margin before your target and discarding it
#    is up to you. :class:`~torchcodec.decoders.AudioDecoder` does all of this
#    for you.
