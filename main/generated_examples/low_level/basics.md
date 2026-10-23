# Build your own decoding pipeline

Important

**The low-level APIs are in beta.** Their signatures and semantics may still
change slightly, in response to user feedback. Please [share your feedback](https://github.com/meta-pytorch/torchcodec/issues?q=is:open+is:issue)!

In this tutorial, we'll take a tour of the low-level decoding APIs: the three
decoding stages for video and audio, following several streams of a container at
once, seeking, scanning, what the metadata means, and decoding a source that
never ends.

For simple end-to-end decoding, [`VideoDecoder`](../../generated/torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder) and
[`AudioDecoder`](../../generated/torchcodec.decoders.AudioDecoder.html#torchcodec.decoders.AudioDecoder) are often all you need: they
handle demuxing, decoding and conversion for you, in a single call. But that
means you don't control these stages: you can't run them on different threads,
stop before the conversion, or decode several streams in a single pass. The
low-level APIs expose those three stages separately, one chain per media type:

```
Demuxer -> VideoPacketDecoder -> ColorConverter
 Packet RawFrame RGB Frame

Demuxer -> AudioPacketDecoder -> AudioConverter
 Packet RawAudioSamples AudioSamples
```

This unlocks features that the high-level decoders don't offer:

- **Performance gains via multi-threaded pipelines**: demux, decode and color-convert on separate
threads. Each stage releases the GIL. ([tutorial](pipelines.html#sphx-glr-generated-examples-low-level-pipelines-py)).
- **Access raw YUV data, for SDR and HDR sources**: read the decoder's own planes, with no
conversion and no copy, at the source's own precision. 10-bit HDR comes out
as `uint16` with every bit intact. You also get access to the raw audio samples.
([tutorial](raw_data.html#sphx-glr-generated-examples-low-level-raw-data-py)).
- **Custom transformations of YUV data**: write your own color conversion kernel, or
train directly in YUV space! ([tutorial](raw_data.html#raw-data-custom-conversion)).
- **Multi-stream decoding**: decode audio and video (or several streams)
in a single pass over the input (below).
- **Endless streams**: decode sources that have no duration, no frame count
and no end, such as live streams and pipes
(below).
- **Key frame retrieval**: get exact key frame positions from a scan, and
decode key frames for cheap thumbnails or training samplers
(below).

First, a bit of boilerplate: a test video, and the device we'll run on.

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
 "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=5",
 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "30",
 "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
 str(video_path),
 ],
 check=True,
)
```

```
device = 'cuda'

CompletedProcess(args=['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=1280x720:rate=30:duration=5', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-g', '30', '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '/tmp/tmppeepq133/video.mp4'], returncode=0)
```

## The three stages

### One video stream

Here is a complete video pipeline, which is equivalent to
[`VideoDecoder.get_all_frames()`](../../generated/torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder.get_all_frames)

```
from torchcodec.decoders import ColorConverter, Demuxer

demuxer = Demuxer(video_path)
(video_stream,) = demuxer.streams
packet_decoder = video_stream.make_decoder(device=device)
color_converter = ColorConverter(device=device)

def decode_frames(demuxer, packet_decoder, color_converter):
 for packet in demuxer:
 for raw_frame in packet_decoder.decode(packet):
 yield color_converter.convert(raw_frame)
 for raw_frame in packet_decoder.drain():
 yield color_converter.convert(raw_frame)

frames = list(decode_frames(demuxer, packet_decoder, color_converter))
print(f"{len(frames)} frames, {frames[0].data.shape = }, "
 f"{frames[0].pts_seconds = }, {frames[0].data.device = }")
```

```
150 frames, frames[0].data.shape = torch.Size([3, 720, 1280]), frames[0].pts_seconds = 0.0, frames[0].data.device = device(type='cuda', index=0)
```

We create a [`Demuxer`](../../generated/torchcodec.decoders.Demuxer.html#torchcodec.decoders.Demuxer) over the file
and let it follow the default video stream, which comes back as a
[`VideoStream`](../../generated/torchcodec.decoders.VideoStream.html#torchcodec.decoders.VideoStream). From that stream we build a [`VideoPacketDecoder`](../../generated/torchcodec.decoders.VideoPacketDecoder.html#torchcodec.decoders.VideoPacketDecoder),
which decodes the demuxer's [`Packet`](../../generated/torchcodec.decoders.Packet.html#torchcodec.decoders.Packet) objects into [`RawFrame`](../../generated/torchcodec.decoders.RawFrame.html#torchcodec.decoders.RawFrame)
objects. A [`RawFrame`](../../generated/torchcodec.decoders.RawFrame.html#torchcodec.decoders.RawFrame) typically contains raw YUV data that you can
access: see [Raw frames and raw audio samples](raw_data.html#sphx-glr-generated-examples-low-level-raw-data-py) for more
details. Finally, a [`ColorConverter`](../../generated/torchcodec.decoders.ColorConverter.html#torchcodec.decoders.ColorConverter) turns
each of those into an RGB [`Frame`](../../generated/torchcodec.Frame.html#torchcodec.Frame), the same object a
[`VideoDecoder`](../../generated/torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder) would have handed you.

Importantly, [`VideoPacketDecoder.decode()`](../../generated/torchcodec.decoders.VideoPacketDecoder.html#torchcodec.decoders.VideoPacketDecoder.decode) returns a list of
[`RawFrame`](../../generated/torchcodec.decoders.RawFrame.html#torchcodec.decoders.RawFrame) objects: depending on how the video was encoded, a single
packet may contain a single frame, several frames, or even a partial frame.
That's why the returned list can be empty, or contain more than one frame, and
you must iterate over it. For the same reason, you must call
[`drain()`](../../generated/torchcodec.decoders.VideoPacketDecoder.html#torchcodec.decoders.VideoPacketDecoder.drain) at the end to retrieve any remaining frames
that the decoder may be holding.

Both [`VideoStream.make_decoder()`](../../generated/torchcodec.decoders.VideoStream.html#torchcodec.decoders.VideoStream.make_decoder) and [`ColorConverter`](../../generated/torchcodec.decoders.ColorConverter.html#torchcodec.decoders.ColorConverter) accept a
`device` parameter so you can run them on CPU or CUDA (they must match!),
and the demuxing stage is always CPU only.

### One audio stream

Audio decoding has the same three stages. The following pipeline is equivalent
to [`AudioDecoder.get_all_samples()`](../../generated/torchcodec.decoders.AudioDecoder.html#torchcodec.decoders.AudioDecoder.get_all_samples):

```
from torchcodec.decoders import AudioConverter

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
audio_converter = AudioConverter(sample_rate=16_000, num_channels=1)

samples = []
for packet in demuxer:
 samples += [audio_converter.convert(raw) for raw in packet_decoder.decode(packet)]
samples += [audio_converter.convert(raw) for raw in packet_decoder.drain()]
samples.append(audio_converter.drain()) # don't forget me

data = torch.cat([s.data for s in samples], dim=1)
print(f"{data.shape = }, {data.dtype = }, "
 f"{samples[-1].data.shape[1]} samples came out of drain()")
```

```
data.shape = torch.Size([1, 80000]), data.dtype = torch.float32, 16 samples came out of drain()
```

An [`AudioStream`](../../generated/torchcodec.decoders.AudioStream.html#torchcodec.decoders.AudioStream) builds an [`AudioPacketDecoder`](../../generated/torchcodec.decoders.AudioPacketDecoder.html#torchcodec.decoders.AudioPacketDecoder), which decodes
[`Packet`](../../generated/torchcodec.decoders.Packet.html#torchcodec.decoders.Packet) objects into [`RawAudioSamples`](../../generated/torchcodec.decoders.RawAudioSamples.html#torchcodec.decoders.RawAudioSamples): the codec's own
samples, in the codec's own sample type, as a `[num_channels, num_samples]`
tensor. Those are the true source samples - 16-bit integers for the file
above, not floats in `[-1, 1]`. [`AudioConverter`](../../generated/torchcodec.decoders.AudioConverter.html#torchcodec.decoders.AudioConverter) is what normalizes
them, and it can resample and change the channel count on the way.

The [`AudioConverter`](../../generated/torchcodec.decoders.AudioConverter.html#torchcodec.decoders.AudioConverter) differs from [`ColorConverter`](../../generated/torchcodec.decoders.ColorConverter.html#torchcodec.decoders.ColorConverter) in one
important way: it is a stateful stream processor (much like the
[`VideoPacketDecoder`](../../generated/torchcodec.decoders.VideoPacketDecoder.html#torchcodec.decoders.VideoPacketDecoder) and [`AudioPacketDecoder`](../../generated/torchcodec.decoders.AudioPacketDecoder.html#torchcodec.decoders.AudioPacketDecoder) are), while the
color converter is mainly stateless. The reason is that resampling is an
interpolation filter, so the sample it emits at a given instant depends on
input samples before and after it. The converter therefore holds the tail
of each [`RawAudioSamples`](../../generated/torchcodec.decoders.RawAudioSamples.html#torchcodec.decoders.RawAudioSamples) until the next one arrives, so that it can
correctly interpolate them. Make sure to call [`AudioConverter.drain()`](../../generated/torchcodec.decoders.AudioConverter.html#torchcodec.decoders.AudioConverter.drain) at
the end of the pipeline to retrieve the last samples.

### Several streams at once

The examples above followed a single stream each. But a [`Demuxer`](../../generated/torchcodec.decoders.Demuxer.html#torchcodec.decoders.Demuxer) can
follow multiple video streams, multiple audio streams, or a mix of both at
once!

```
av_path = temp_dir / "av.mp4"
subprocess.run(
 [
 "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
 "-i", str(video_path), "-i", str(audio_path),
 "-c:v", "copy", "-c:a", "aac", "-shortest", str(av_path),
 ],
 check=True,
)
```

```
CompletedProcess(args=['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-i', '/tmp/tmppeepq133/video.mp4', '-i', '/tmp/tmppeepq133/audio.wav', '-c:v', 'copy', '-c:a', 'aac', '-shortest', '/tmp/tmppeepq133/av.mp4'], returncode=0)
```

Which streams to follow is specified at construction time of the
[`Demuxer`](../../generated/torchcodec.decoders.Demuxer.html#torchcodec.decoders.Demuxer) via the `streams` parameter. `streams` takes one selector
or a tuple of them: `"video"` or `"audio"` for the [best stream](../../glossary.html#term-best-stream) of
that type, an `int` for a stream index - which is how you reach the second
video stream, or the fourth audio track - or `"all"` on its own for every
audio and video stream in the file. `demuxer.streams` comes back in the
order you asked for, and each of them builds its own decoder.

```
demuxer = Demuxer(av_path, streams=("video", "audio"))
video_stream, audio_stream = demuxer.streams

decoders = {
 video_stream.index: video_stream.make_decoder(device=device),
 audio_stream.index: audio_stream.make_decoder(),
}
color_converter = ColorConverter(device=device)
audio_converter = AudioConverter()
```

The packets come out interleaved, in the order the container stores them, and
[`Packet.stream_index`](../../generated/torchcodec.decoders.Packet.html#torchcodec.decoders.Packet.stream_index) says which stream each belongs to: use it to match
a packet to its decoder. Each decoder is drained at the end, as always.

```
frames, samples = [], []
for packet in demuxer:
 outputs = decoders[packet.stream_index].decode(packet)
 if packet.stream_index == video_stream.index:
 frames += [color_converter.convert(raw) for raw in outputs]
 else:
 samples += [audio_converter.convert(raw) for raw in outputs]

frames += [color_converter.convert(raw)
 for raw in decoders[video_stream.index].drain()]
samples += [audio_converter.convert(raw)
 for raw in decoders[audio_stream.index].drain()]
samples.append(audio_converter.drain())

data = torch.cat([s.data for s in samples], dim=1)
print(f"{len(frames)} frames up to {frames[-1].pts_seconds:.2f}s, "
 f"{data.shape[1]} samples in one pass over the file")
```

```
150 frames up to 4.97s, 221184 samples in one pass over the file
```

## Seeking

[`Demuxer.seek()`](../../generated/torchcodec.decoders.Demuxer.html#torchcodec.decoders.Demuxer.seek) moves the demuxer to a timestamp. For videos, a decoder
can only start on a keyframe, so the seek lands on the keyframe at or before
the target, and the first frames that come out usually precede it: keep
decoding forward and drop them until you reach the timestamp you asked for.

The seek also invalidates the frames the decoder is holding on to, so the
[`VideoPacketDecoder`](../../generated/torchcodec.decoders.VideoPacketDecoder.html#torchcodec.decoders.VideoPacketDecoder) must be [`reset()`](../../generated/torchcodec.decoders.VideoPacketDecoder.html#torchcodec.decoders.VideoPacketDecoder.reset). So must
every other decoder fed by that demuxer, and every [`AudioConverter`](../../generated/torchcodec.decoders.AudioConverter.html#torchcodec.decoders.AudioConverter),
whose resampler carries state across calls just the same.

```
demuxer = Demuxer(video_path)
packet_decoder = demuxer.streams[0].make_decoder(device=device)
color_converter = ColorConverter(device=device)

seconds = 2.5
demuxer.seek(seconds)
packet_decoder.reset() # An error is raised if you forget this!

frames = decode_frames(demuxer, packet_decoder, color_converter)
landed_on = next(frames)
# The frame *playing* at a timestamp is the first one that hasn't finished
# playing by then.
target = next(
 frame
 for frame in frames
 if frame.pts_seconds + frame.duration_seconds > seconds
)
print(f"asked for {seconds}s, landed on {landed_on.pts_seconds:.3f}s, "
 f"target frame at {target.pts_seconds:.3f}s")
```

```
asked for 2.5s, landed on 2.000s, target frame at 2.500s
```

That is what [`VideoDecoder.get_frame_played_at()`](../../generated/torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder.get_frame_played_at) does for you - the
seek, the decoding forward, and the dropping - except that we did it in the
decoder's `seek_mode="approximate"` flavour. The `"exact"` one needs a
[scan](../../glossary.html#term-scan), which the next section gets to.

Warning

Important for audio: these APIs do no pre-roll. A lossy codec's first
frames after a seek are subtly wrong until it re-primes, especially when
resampling is involved. Decoding a margin before and after your target and
discarding it is up to you. [`AudioDecoder`](../../generated/torchcodec.decoders.AudioDecoder.html#torchcodec.decoders.AudioDecoder) does
all of this for you.

## Scanning

[`VideoStream.scan()`](../../generated/torchcodec.decoders.VideoStream.html#torchcodec.decoders.VideoStream.scan) demuxes the whole stream once, without decoding
anything, and returns a [`FrameIndex`](../../generated/torchcodec.decoders.FrameIndex.html#torchcodec.decoders.FrameIndex): one entry per frame, in
presentation order.
This is the only way to know a stream's exact frame count, timestamps and
keyframe positions - a container header can be wrong about all of them. It
costs one pass over the file, and it leaves the demuxer back at the start.

If you call [`VideoStream.scan()`](../../generated/torchcodec.decoders.VideoStream.html#torchcodec.decoders.VideoStream.scan), it has to happen before any packet is
read from the demuxer.

```
demuxer = Demuxer(video_path)
(video_stream,) = demuxer.streams
index = video_stream.scan()
packet_decoder = video_stream.make_decoder(device=device)
color_converter = ColorConverter(device=device)

print(f"{index.num_frames_from_content} frames at "
 f"{index.average_fps_from_content} fps, "
 f"from {index.begin_stream_seconds_from_content}s "
 f"to {index.end_stream_seconds_from_content}s")
```

```
150 frames at 30.0 fps, from 0.0s to 5.0s
```

The index is also what gives the stages frame *indices*, which they otherwise
don't have at all: [`FrameIndex.index_at()`](../../generated/torchcodec.decoders.FrameIndex.html#torchcodec.decoders.FrameIndex.index_at) maps a timestamp to the frame
on screen then, and [`FrameIndex.pts_seconds`](../../generated/torchcodec.decoders.FrameIndex.html#torchcodec.decoders.FrameIndex.pts_seconds) maps back. That's enough to
build [`get_frame_at()`](../../generated/torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder.get_frame_at), or a clip
sampler, on top of the low-level APIs.

```
i = index.index_at(seconds)
print(f"frame {i} is on screen at {seconds}s, and starts at {index.pts_seconds[i]}s")
```

```
frame 75 is on screen at 2.5s, and starts at 2.5s
```

If you're following more than one video stream, you can call `scan()` on
each of them. You only pay the scan cost once, for the first stream: the other
streams' scan resuts are cached and returned when you call scan on them.

### Keyframes

[`FrameIndex.is_key_frame`](../../generated/torchcodec.decoders.FrameIndex.html#torchcodec.decoders.FrameIndex.is_key_frame) is a mask over all the frames, and
[`FrameIndex.key_frame_indices`](../../generated/torchcodec.decoders.FrameIndex.html#torchcodec.decoders.FrameIndex.key_frame_indices) the same thing as a list of indices.

A keyframe is the cheapest kind of frame to decode on its own, because every
non-keyframe needs a keyframe to be decoded first before it can itself be
decoded. Reaching an arbitrary frame therefore means seeking to the keyframe
before it and decoding everything in between, while reaching a keyframe costs
exactly one frame: the seek lands right on it, and it is the very next frame
out of the decoder - nothing to decode forward, nothing to drop.

That makes keyframes attractive whenever you need *some* frames of a video
rather than specific ones - thumbnails, coarse previews, or a cheap sampler
for training.

```
key_frame_indices = index.key_frame_indices
print(f"{len(key_frame_indices)} keyframes out of {len(index)} frames, "
 f"at {index.pts_seconds[key_frame_indices].tolist()}")

def keyframe_sampler(demuxer, packet_decoder, color_converter, index):
 for k in index.key_frame_indices.tolist():
 demuxer.seek(index.pts_seconds[k])
 packet_decoder.reset()
 # The seek landed on the keyframe itself, so the first frame out of the
 # decoder is the one we want.
 yield next(decode_frames(demuxer, packet_decoder, color_converter))

key_frames = list(
 keyframe_sampler(demuxer, packet_decoder, color_converter, index)
)
print(f"decoded {len(key_frames)} keyframes at "
 f"{[round(f.pts_seconds, 3) for f in key_frames]}")
```

```
5 keyframes out of 150 frames, at [0.0, 1.0, 2.0, 3.0, 4.0]
decoded 5 keyframes at [0.0, 1.0, 2.0, 3.0, 4.0]
```

### Exact seeking

One last thing the index buys you, which most files never need.
`seek(seconds)` already lands on the keyframe at or before the target - but
without a scan, FFmpeg resolves a seek against metadata that might not be
fully accurate. In rare cases, you might land on a keyframe that is *after*
the one you wanted. [`FrameIndex.key_frame_seconds_for()`](../../generated/torchcodec.decoders.FrameIndex.html#torchcodec.decoders.FrameIndex.key_frame_seconds_for) gives you that
keyframe's own timestamp instead, which always lands where you meant.

The two are the same call on the majority of files. This is what separates
[`VideoDecoder`](../../generated/torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder)'s `seek_mode="exact"` from
`"approximate"` for a timestamp lookup.

```
demuxer.seek(index.key_frame_seconds_for(seconds))
packet_decoder.reset()

frames = decode_frames(demuxer, packet_decoder, color_converter)
target = next(f for f in frames if f.pts_seconds >= index.pts_seconds[i])
print(f"frame {i} at {target.pts_seconds:.3f}s")
```

```
frame 75 at 2.500s
```

## Metadata

Metadata comes in three tiers, and the low-level APIs never merge them:

| Where | Type | What it describes |
| --- | --- | --- |
| `demuxer.metadata` | [`DemuxerMetadata`](../../generated/torchcodec.decoders.DemuxerMetadata.html#torchcodec.decoders.DemuxerMetadata) | the container |
| `stream.metadata` | [`VideoStreamHeaderMetadata`](../../generated/torchcodec.decoders.VideoStreamHeaderMetadata.html#torchcodec.decoders.VideoStreamHeaderMetadata) or [`AudioStreamHeaderMetadata`](../../generated/torchcodec.decoders.AudioStreamHeaderMetadata.html#torchcodec.decoders.AudioStreamHeaderMetadata) | what the header claims about one stream |
| `video_stream.scan()` | [`FrameIndex`](../../generated/torchcodec.decoders.FrameIndex.html#torchcodec.decoders.FrameIndex) | what that stream's packets actually say |

The name of a field tells you which tier it came from, so a header value is
never mistaken for an exact one:

```
demuxer = Demuxer(video_path)
(video,) = demuxer.streams

print(f"container: {demuxer.metadata.duration_seconds_from_header}s")
print(f"header: {video.metadata.width}x{video.metadata.height}, "
 f"{video.metadata.num_frames_from_header} frames "
 f"at {video.metadata.average_fps_from_header} fps")
print(f"content: {video.scan().num_frames_from_content} frames "
 f"at {video.scan().average_fps_from_content} fps")
```

```
container: 5s
header: 1280x720, 150 frames at 30 fps
content: 150 frames at 30.0 fps
```

### Comparing with VideoDecoder and AudioDecoder

A [`VideoDecoder`](../../generated/torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder)'s `metadata` is a
[`VideoStreamMetadata`](../../generated/torchcodec.decoders.VideoStreamMetadata.html#torchcodec.decoders.VideoStreamMetadata), which holds two kinds of
fields:

- The **raw** ones, suffixed `_from_header` or `_from_content`. Every one
of them is available from the low-level APIs too, on `demuxer.metadata`,
on
`stream.metadata`, or on the [`FrameIndex`](../../generated/torchcodec.decoders.FrameIndex.html#torchcodec.decoders.FrameIndex) a [scan](../../glossary.html#term-scan) returns.
Same values, same names.
- A fourth, **"magic"** set with no suffix - `num_frames`,
`average_fps`, `duration_seconds`, `begin_stream_seconds`,
`end_stream_seconds`. These don't exist in the file: each one runs a
*fallback chain* over the raw fields and hands you the first thing that
isn't `None`.

**The low-level APIs run no fallback logic at all**, so the magic set has no
low-level equivalent. What you get instead is every input those chains read
from, which you are free to combine yourself:

| [`VideoDecoder.metadata`](../../generated/torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder) | Low-level equivalent |
| --- | --- |
| `width`, `height`, `codec`, ... | `stream.metadata.<same name>` |
| `num_frames_from_header` | `stream.metadata.num_frames_from_header` |
| `num_frames_from_content` | `stream.scan().num_frames_from_content` |
| `num_frames` | *(none)* - `stream.scan().num_frames_from_content` if scanned, else `stream.metadata.num_frames_from_header`, else computed from `duration_seconds` and `average_fps` |
| `average_fps` | *(none)* - `stream.scan().average_fps_from_content` if scanned, else `stream.metadata.average_fps_from_header` |
| `duration_seconds` | *(none)* - from the scan if scanned, else `stream.metadata.duration_seconds_from_header`, else computed from `stream.metadata.num_frames_from_header` and `stream.metadata.average_fps_from_header`, else `demuxer.metadata.duration_seconds_from_header` |
| `begin_stream_seconds` | *(none)* - `stream.scan().begin_stream_seconds_from_content` if scanned, else `0` |
| `end_stream_seconds` | *(none)* - `stream.scan().end_stream_seconds_from_content` if scanned, else `duration_seconds` |

Note how often "if scanned" shows up. Whether a
[`VideoDecoder`](../../generated/torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder) scanned is a consequence of its
`seek_mode`: `"exact"` (the default) scans up-front, `"approximate"`
never does. So `metadata.num_frames` silently means a different thing
depending on how the decoder was built, and that is exactly the ambiguity the
low-level APIs refuse to introduce.

[`AudioDecoder`](../../generated/torchcodec.decoders.AudioDecoder.html#torchcodec.decoders.AudioDecoder) works the same way, with a smaller
magic set on [`AudioStreamMetadata`](../../generated/torchcodec.decoders.AudioStreamMetadata.html#torchcodec.decoders.AudioStreamMetadata):
`sample_rate`, `num_channels` and `sample_format` are raw header fields
and are on [`AudioStream.metadata`](../../generated/torchcodec.decoders.AudioStream.html#torchcodec.decoders.AudioStream.metadata) too, while `duration_seconds` and
`begin_stream_seconds` are computed.

### Reading metadata of a file before opening it

A demuxer's streams are fixed when you construct it, so to decide which ones
to ask for you need to know what is in the file first. That's
[`get_container_metadata()`](../../generated/torchcodec.decoders.get_container_metadata.html#torchcodec.decoders.get_container_metadata): it reads the header and no packets, and it
also reports the streams a demuxer cannot follow, such as subtitles. It hands
back a [`ContainerMetadata`](../../generated/torchcodec.decoders.ContainerMetadata.html#torchcodec.decoders.ContainerMetadata).

```
from torchcodec.decoders import get_container_metadata

for stream in get_container_metadata(av_path).streams:
 print(f" stream {stream.stream_index}: {stream.media_type}, {stream.codec}")
```

```
stream 0: video, h264
stream 1: audio, aac
```

## Streams of unknown length

One last thing the low-level APIs make possible.
[`VideoDecoder`](../../generated/torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder) needs a finite, seekable source: it
relies on the stream's duration and frame count, and in its default
`seek_mode="exact"` it scans the whole file up-front. The stages never do
that: they consume packets as they arrive, so they can decode a source with no
duration, no frame count, and no end.

Here is one such example where we generate an endless stream with FFmpeg, and
decode its first 100 frames:

```
import os

fifo_path = temp_dir / "live.ts"
os.mkfifo(fifo_path)
ffmpeg = subprocess.Popen(
 [
 "ffmpeg", "-hide_banner", "-loglevel", "error",
 "-f", "lavfi", "-i", "testsrc2=size=640x480:rate=30", # no duration!
 "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
 "-g", "30", "-f", "mpegts", "-y", str(fifo_path),
 ],
)

demuxer = Demuxer(fifo_path)
packet_decoder = demuxer.streams[0].make_decoder(device=device)
color_converter = ColorConverter(device=device)

frames = []
for frame in decode_frames(demuxer, packet_decoder, color_converter):
 frames.append(frame)
 if len(frames) == 100:
 break

print(f"{len(frames)} frames, from pts {frames[0].pts_seconds:.2f}s "
 f"to {frames[-1].pts_seconds:.2f}s")

ffmpeg.kill()
ffmpeg.wait()
```

```
100 frames, from pts 1.40s to 4.70s

-9
```

## Where to go next

- [Multi-threaded decoding pipelines](pipelines.html#sphx-glr-generated-examples-low-level-pipelines-py) runs the stages
concurrently on several threads, and shows where to split a pipeline on CPU
and on CUDA.
- [Raw frames and raw audio samples](raw_data.html#sphx-glr-generated-examples-low-level-raw-data-py) skips the converters
and reads the decoder's own YUV planes and audio samples, at the source's
own precision.
- [Low-level APIs and CUDA streams synchronization](cuda_streams.html#sphx-glr-generated-examples-low-level-cuda-streams-py) explains how to
manage CUDA streams when you consume a [`RawFrame`](../../generated/torchcodec.decoders.RawFrame.html#torchcodec.decoders.RawFrame) on a different CUDA
stream than the one it was decoded on.

**Total running time of the script:** (0 minutes 1.522 seconds)

[`Download Jupyter notebook: basics.ipynb`](../../_downloads/6982b776b242e8d98bf3991f1eabc245/basics.ipynb)

[`Download Python source code: basics.py`](../../_downloads/b70ef813f0b02fec274129a39328a16b/basics.py)

[`Download zipped: basics.zip`](../../_downloads/882ebfafbb55618db26928e2f873c88a/basics.zip)

[Gallery generated by Sphinx-Gallery](https://sphinx-gallery.github.io)