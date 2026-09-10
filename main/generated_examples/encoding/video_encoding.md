# Encoding video with the Encoder

In this example, we'll learn how to encode video frames using the
[`Encoder`](../../generated/torchcodec.encoders.Encoder.html#torchcodec.encoders.Encoder) class, and we'll explore the different
encoding parameters available.

Tip

[`VideoEncoder`](../../generated/torchcodec.encoders.VideoEncoder.html#torchcodec.encoders.VideoEncoder) and
[`AudioEncoder`](../../generated/torchcodec.encoders.AudioEncoder.html#torchcodec.encoders.AudioEncoder) are simpler, single-stream
encoders. They are convenient for simple one-shot encoding, but the
[`Encoder`](../../generated/torchcodec.encoders.Encoder.html#torchcodec.encoders.Encoder) class is more flexible and supports
the same encoding options. See
[Encoding audio and video streams with the Encoder](multi_stream_encoding.html#sphx-glr-generated-examples-encoding-multi-stream-encoding-py) for a
multi-stream encoding tutorial.

First, let's download a video and decode some frames to tensors.
These will be the input for our encoder. For more details on decoding,
see [Decoding a video with VideoDecoder](../decoding/basic_example.html#sphx-glr-generated-examples-decoding-basic-example-py).
Otherwise, skip ahead to Creating an encoder and encoding to a file.

```
import io

import requests
from torchcodec.decoders import VideoDecoder
from IPython.display import Video

def play_video(encoded_bytes):
 return Video(
 data=encoded_bytes,
 embed=True,
 width=640,
 height=360,
 mimetype="video/mp4",
 )

# Video source: https://www.pexels.com/video/adorable-cats-on-the-lawn-4977395/
# Author: Altaf Shah.
url = "https://videos.pexels.com/video-files/4977395/4977395-hd_1920_1080_24fps.mp4"

response = requests.get(url, headers={"User-Agent": ""})
if response.status_code != 200:
 raise RuntimeError(f"Failed to download video. {response.status_code = }.")

raw_video_bytes = response.content

decoder = VideoDecoder(raw_video_bytes)
frames = decoder.get_frames_in_range(0, 60).data # Get first 60 frames
frame_rate = decoder.metadata.average_fps
```

## Creating an encoder and encoding to a file

Let's instantiate an [`Encoder`](../../generated/torchcodec.encoders.Encoder.html#torchcodec.encoders.Encoder), add a video
stream, and encode to a file. We use the
[`open_file()`](../../generated/torchcodec.encoders.Encoder.html#torchcodec.encoders.Encoder.open_file) method as a context manager to
ensure the output is properly flushed and closed.

Note

The `frame_rate` parameter corresponds to the frame rate of the
*input* video. It will also be used for the frame rate of the *output*
encoded video.

```
import tempfile
from pathlib import Path
from torchcodec.encoders import Encoder

print(f"{frames.shape = }, {frames.dtype = }")
print(f"{frame_rate = } fps")

output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
encoder = Encoder()
video_stream = encoder.add_video(
 height=frames.shape[2], width=frames.shape[3], frame_rate=frame_rate
)
with encoder.open_file(output_path):
 video_stream.add_frames(frames)
 # More frames can be submitted by calling video_stream.add_frames

print(f"Encoded to {output_path}, size: {Path(output_path).stat().st_size} bytes")
```

```
frames.shape = torch.Size([60, 3, 1080, 1920]), frames.dtype = torch.uint8
frame_rate = 24 fps
Encoded to /tmp/tmpc4ta4k3b.mp4, size: 2509386 bytes
```

Now that we have encoded data, let's decode it back to verify the
round-trip encode/decode process works as expected:

```
decoder_verify = VideoDecoder(output_path)
decoded_frames = decoder_verify.get_frames_in_range(0, 60).data

print(f"Re-decoded video: {decoded_frames.shape = }")
print(f"Original frames: {frames.shape = }")
```

```
Re-decoded video: decoded_frames.shape = torch.Size([60, 3, 1080, 1920])
Original frames: frames.shape = torch.Size([60, 3, 1080, 1920])
```

## CUDA Encoding

To encode on GPU, pass `device="cuda"` to
[`add_video()`](../../generated/torchcodec.encoders.Encoder.html#torchcodec.encoders.Encoder.add_video), and feed CUDA tensors to
[`add_frames()`](../../generated/torchcodec.encoders.VideoStream.html#torchcodec.encoders.VideoStream.add_frames). This can result in
significantly faster encoding than CPU. The encoder will automatically select
a CUDA-compatible codec like `h264_nvenc` or `hevc_nvenc`.

Note

On GPU, the pixel format is always set to `nv12` (which does equivalent
chroma subsampling to `yuv420p`). The `pixel_format` parameter is not
supported for GPU encoding.

```
gpu_frames = frames.to("cuda") # Move frames to GPU
encoder = Encoder()
video_stream = encoder.add_video(
 height=gpu_frames.shape[2], width=gpu_frames.shape[3],
 frame_rate=frame_rate, device="cuda",
)
with encoder.open_file("output.mp4"):
 video_stream.add_frames(gpu_frames)
```

That's it! The rest of the encoding process is the same as on CPU.

## Codec Selection

By default, the codec is selected automatically based on the container format.
For example, when encoding to MP4 format, the default codec is typically
`H.264`.

To use a codec other than the default, use the `codec` parameter in
[`add_video()`](../../generated/torchcodec.encoders.Encoder.html#torchcodec.encoders.Encoder.add_video). You can specify either a
specific codec implementation (e.g., `"libx264"`) or a codec specification
(e.g., `"h264"`). Different codecs offer different tradeoffs between
quality, file size, and encoding speed.

Note

To see available encoders on your system, run `ffmpeg -encoders`.

Let's encode the same frames using different codecs:

```
# H.264 encoding
h264_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
encoder = Encoder()
vs = encoder.add_video(
 height=frames.shape[2], width=frames.shape[3],
 frame_rate=frame_rate, codec="libx264",
)
with encoder.open_file(h264_output):
 vs.add_frames(frames)

# H.265 encoding
hevc_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
encoder = Encoder()
vs = encoder.add_video(
 height=frames.shape[2], width=frames.shape[3],
 frame_rate=frame_rate, codec="hevc",
)
with encoder.open_file(hevc_output):
 vs.add_frames(frames)

# Now let's use ffprobe to verify the codec used in the output files
import subprocess

for output, name in [(h264_output, "h264_output"), (hevc_output, "hevc_output")]:
 result = subprocess.run(
 [
 "ffprobe",
 "-v",
 "error",
 "-select_streams",
 "v:0",
 "-show_entries",
 "stream=codec_name",
 "-of",
 "default=noprint_wrappers=1:nokey=1",
 output,
 ],
 capture_output=True,
 text=True,
 )
 print(f"Codec used in {name}: {result.stdout.strip()}")
```

```
Codec used in h264_output: h264
Codec used in hevc_output: hevc
```

## Pixel Format

The `pixel_format` parameter controls the color sampling (chroma subsampling)
of the output video. This affects both quality and file size.

Common pixel formats:

- `"yuv420p"` - 4:2:0 chroma subsampling (standard quality, smaller file size, widely compatible)
- `"yuv444p"` - 4:4:4 chroma subsampling (full chroma resolution, higher quality, larger file size)

Most playback devices and platforms support `yuv420p`, making it the most
common choice for video encoding.

Note

Pixel format support depends on the codec used. Use `ffmpeg -h encoder=<codec_name>`
to check available options for your selected codec.

```
# Standard pixel format
buf = io.BytesIO()
encoder = Encoder()
vs = encoder.add_video(
 height=frames.shape[2], width=frames.shape[3],
 frame_rate=frame_rate, codec="libx264", pixel_format="yuv420p",
)
with encoder.open_file_like(buf, format="mp4"):
 vs.add_frames(frames)

play_video(buf.getvalue())
```

 Your browser does not support the video tag.
 

## CRF (Constant Rate Factor)

The `crf` parameter controls video quality, where lower values produce higher quality output.

For example, with the commonly used H.264 codec, `libx264`:

- Values range from 0 (lossless) to 51 (worst quality)
- Values 17 or 18 are considered visually lossless, and the default is 23.

Note

The range and interpretation of CRF values depend on the codec used, and
not all codecs support CRF. Use `ffmpeg -h encoder=<codec_name>` to
check available options for your selected codec.

```
# High quality (low CRF)
buf = io.BytesIO()
encoder = Encoder()
vs = encoder.add_video(
 height=frames.shape[2], width=frames.shape[3],
 frame_rate=frame_rate, codec="libx264", crf=0,
)
with encoder.open_file_like(buf, format="mp4"):
 vs.add_frames(frames)

# play_video is disabled because crf=0 creates a 50+ Mb video that we don't want
# to check into our docs
# play_video(buf.getvalue())
```

```
# Low quality (high CRF)
buf = io.BytesIO()
encoder = Encoder()
vs = encoder.add_video(
 height=frames.shape[2], width=frames.shape[3],
 frame_rate=frame_rate, codec="libx264", crf=50,
)
with encoder.open_file_like(buf, format="mp4"):
 vs.add_frames(frames)

play_video(buf.getvalue())
```

 Your browser does not support the video tag.
 

## Preset

The `preset` parameter controls the tradeoff between encoding speed and file compression.
Faster presets encode faster but produce larger files, while slower
presets take more time to encode but result in better compression.

For example, with the commonly used H.264 codec, `libx264` presets include
`"ultrafast"` (fastest), `"fast"`, `"medium"` (default), `"slow"`, and
`"veryslow"` (slowest, best compression). See the
[H.264 Video Encoding Guide](https://trac.ffmpeg.org/wiki/Encode/H.264#a2.Chooseapresetandtune)
for additional details.

Note

Not all codecs support the `presets` option. Use `ffmpeg -h encoder=<codec_name>`
to check available options for your selected codec.

```
# Fast encoding with a larger file size
fast_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
encoder = Encoder()
vs = encoder.add_video(
 height=frames.shape[2], width=frames.shape[3],
 frame_rate=frame_rate, codec="libx264", preset="ultrafast",
)
with encoder.open_file(fast_output):
 vs.add_frames(frames)

print(f"Size of fast encoded file: {Path(fast_output).stat().st_size} bytes")

# Slow encoding for a smaller file size
slow_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
encoder = Encoder()
vs = encoder.add_video(
 height=frames.shape[2], width=frames.shape[3],
 frame_rate=frame_rate, codec="libx264", preset="veryslow",
)
with encoder.open_file(slow_output):
 vs.add_frames(frames)

print(f"Size of slow encoded file: {Path(slow_output).stat().st_size} bytes")
```

```
Size of fast encoded file: 7253009 bytes
Size of slow encoded file: 2110051 bytes
```

## Extra Options

The `extra_options` parameter accepts a dictionary of codec-specific options
that would normally be set via FFmpeg command-line arguments. This enables
control of encoding settings beyond the common parameters.

For example, some potential extra options for the commonly used H.264 codec, `libx264` include:

- `"g"` - GOP (Group of Pictures) size / keyframe interval
- `"max_b_frames"` - Maximum number of B-frames between I and P frames
- `"tune"` - Tuning preset (e.g., `"film"`, `"animation"`, `"grain"`)

Note

Use `ffmpeg -h encoder=<codec_name>` to see all available options for
a specific codec.

```
custom_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
encoder = Encoder()
vs = encoder.add_video(
 height=frames.shape[2], width=frames.shape[3],
 frame_rate=frame_rate, codec="libx264",
 extra_options={
 "g": 50, # Keyframe every 50 frames
 "max_b_frames": 0, # Disable B-frames for faster decoding
 "tune": "fastdecode", # Optimize for fast decoding
 },
)
with encoder.open_file(custom_output):
 vs.add_frames(frames)
```

**Total running time of the script:** (0 minutes 26.645 seconds)

[`Download Jupyter notebook: video_encoding.ipynb`](../../_downloads/e8868f804bffbeae56ed6b335353e809/video_encoding.ipynb)

[`Download Python source code: video_encoding.py`](../../_downloads/2da325461f15d27a53d05baddd7a3f9c/video_encoding.py)

[`Download zipped: video_encoding.zip`](../../_downloads/b642c54b41acc0759a9f5f6a35c7bb9a/video_encoding.zip)

[Gallery generated by Sphinx-Gallery](https://sphinx-gallery.github.io)