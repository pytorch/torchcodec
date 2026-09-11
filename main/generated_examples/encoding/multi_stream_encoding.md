# Encoding audio and video streams with the Encoder

In this example, we'll learn how to encode multiple video and audio streams
into a single container using the [`Encoder`](../../generated/torchcodec.encoders.Encoder.html#torchcodec.encoders.Encoder) class.
We'll also see how to feed audio samples and video frames incrementally, and how
to mix CPU and CUDA video streams.

For details on video encoding parameters (codec, CRF, preset, etc.), see
[Encoding video with the Encoder](video_encoding.html#sphx-glr-generated-examples-encoding-video-encoding-py).

## Video + audio encoding

Let's start by encoding a video alongside an audio track into the same MP4
file. We'll decode some video frames from an existing video and generate a
simple sine-wave audio tone.

```
import subprocess
import tempfile
from pathlib import Path

import requests
import torch
from torchcodec.decoders import VideoDecoder
from torchcodec.encoders import Encoder

# Video source: https://www.pexels.com/video/adorable-cats-on-the-lawn-4977395/
# Author: Altaf Shah.
url = "https://videos.pexels.com/video-files/4977395/4977395-hd_1920_1080_24fps.mp4"

response = requests.get(url, headers={"User-Agent": ""})
if response.status_code != 200:
 raise RuntimeError(f"Failed to download video. {response.status_code = }.")

decoder = VideoDecoder(response.content)
frames = decoder.get_frames_in_range(0, 60).data
frame_rate = decoder.metadata.average_fps

# Generate a 440 Hz sine wave that lasts as long as the video
audio_sample_rate = 16000
duration_seconds = len(frames) / frame_rate
t = torch.linspace(
 0, duration_seconds, int(audio_sample_rate * duration_seconds),
 dtype=torch.float32,
)
audio_samples = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0) # shape: (1, num_samples)
```

Now we create an [`Encoder`](../../generated/torchcodec.encoders.Encoder.html#torchcodec.encoders.Encoder), add one video stream
and one audio stream, and encode everything into a single file. Each call to
[`add_video()`](../../generated/torchcodec.encoders.Encoder.html#torchcodec.encoders.Encoder.add_video) or
[`add_audio()`](../../generated/torchcodec.encoders.Encoder.html#torchcodec.encoders.Encoder.add_audio) returns a stream object that
we use to feed data.

```
output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
encoder = Encoder()
video_stream = encoder.add_video(
 height=frames.shape[2], width=frames.shape[3], frame_rate=frame_rate,
)
audio_stream = encoder.add_audio(sample_rate=audio_sample_rate, num_channels=1)

with encoder.open_file(output_path):
 video_stream.add_frames(frames)
 audio_stream.add_samples(audio_samples)

print(f"Encoded video + audio to {output_path}")
print(f"Output size: {Path(output_path).stat().st_size} bytes")
```

```
Encoded video + audio to /tmp/tmp_bu9r6lr.mp4
Output size: 2526674 bytes
```

Let's verify that both streams are present in the output file:

```
result = subprocess.run(
 [
 "ffprobe", "-v", "error",
 "-show_entries", "stream=index,codec_type,codec_name",
 "-of", "default=noprint_wrappers=1", output_path,
 ],
 capture_output=True, text=True,
)
print(result.stdout)
```

```
index=0
codec_name=h264
codec_type=video
index=1
codec_name=aac
codec_type=audio
```

## Incremental encoding

You don't need to have all your data ready upfront. You can call
[`add_frames()`](../../generated/torchcodec.encoders.VideoStream.html#torchcodec.encoders.VideoStream.add_frames) and
[`add_samples()`](../../generated/torchcodec.encoders.AudioStream.html#torchcodec.encoders.AudioStream.add_samples) multiple times to feed
data incrementally. This is useful when frames or samples are generated
on-the-fly (e.g. from a model or a processing pipeline).

Here, we'll split our frames and audio into chunks and feed them one batch at
a time:

```
chunk_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
encoder = Encoder()
video_stream = encoder.add_video(
 height=frames.shape[2], width=frames.shape[3], frame_rate=frame_rate,
)
audio_stream = encoder.add_audio(sample_rate=audio_sample_rate, num_channels=1)

video_chunk_size = 10
samples_per_video_chunk = int(audio_sample_rate / frame_rate * video_chunk_size)

with encoder.open_file(chunk_output):
 for i in range(0, len(frames), video_chunk_size):
 video_chunk = frames[i : i + video_chunk_size]
 video_stream.add_frames(video_chunk)

 audio_start = int(i / frame_rate * audio_sample_rate)
 audio_chunk = audio_samples[:, audio_start : audio_start + samples_per_video_chunk]
 audio_stream.add_samples(audio_chunk)

print(f"Incrementally encoded to {chunk_output}")
print(f"Output size: {Path(chunk_output).stat().st_size} bytes")
```

```
Incrementally encoded to /tmp/tmpi264h85r.mp4
Output size: 2527045 bytes
```

## Multiple video streams, multiple audio streams

You can add as many video and audio streams as you need. Each video stream can
independently target CPU or CUDA encoding -- just pass the desired `device`
to [`add_video()`](../../generated/torchcodec.encoders.Encoder.html#torchcodec.encoders.Encoder.add_video). This means you can mix CPU
and CUDA video streams in the same container, for example encoding a
high-resolution stream on GPU for speed and a low-resolution stream on CPU.

Similarly, you can add multiple audio streams with different settings (sample
rate, number of channels, bit rate, etc.).

Here's an example with two video streams and two audio streams:

```
encoder = Encoder()

# Two video streams: one on CPU, one on CUDA
cpu_video = encoder.add_video(
 height=1080, width=1920, frame_rate=30,
 device="cpu",
)
cuda_video = encoder.add_video(
 height=720, width=1280, frame_rate=30,
 device="cuda",
)

# Two audio streams with different settings
audio_en = encoder.add_audio(sample_rate=44100, num_channels=2)
audio_fr = encoder.add_audio(sample_rate=44100, num_channels=2)

with encoder.open_file("multi_stream_output.mkv"):
 cpu_video.add_frames(cpu_frames)
 cuda_video.add_frames(cuda_frames)
 audio_en.add_samples(english_samples)
 audio_fr.add_samples(french_samples)
```

## Encoding to a file-like object

Instead of encoding to a file path, you can encode to any file-like object
(e.g. `io.BytesIO()`) using
[`open_file_like()`](../../generated/torchcodec.encoders.Encoder.html#torchcodec.encoders.Encoder.open_file_like). This is useful for
example when you need to upload the encoded data directly to a remote server
or cloud storage without writing it to disk. In this case, you must specify
the container `format` explicitly since there is no file extension to infer
it from.

```
import io

buf = io.BytesIO()
encoder = Encoder()
video_stream = encoder.add_video(
 height=frames.shape[2], width=frames.shape[3], frame_rate=frame_rate,
)
audio_stream = encoder.add_audio(sample_rate=audio_sample_rate, num_channels=1)

with encoder.open_file_like(buf, format="mp4"):
 video_stream.add_frames(frames)
 audio_stream.add_samples(audio_samples)

encoded_bytes = buf.getvalue()
print(f"Encoded to BytesIO, size: {len(encoded_bytes)} bytes")

# Or convert to a bytes tensor:
bytes_tensor = torch.frombuffer(encoded_bytes, dtype=torch.uint8)
```

```
Encoded to BytesIO, size: 2526674 bytes
```

**Total running time of the script:** (0 minutes 6.124 seconds)

[`Download Jupyter notebook: multi_stream_encoding.ipynb`](../../_downloads/ccd7596cf7d161af13c32c87cfc7f878/multi_stream_encoding.ipynb)

[`Download Python source code: multi_stream_encoding.py`](../../_downloads/58074f3e1629f5cce08951c5ae1a0c35/multi_stream_encoding.py)

[`Download zipped: multi_stream_encoding.zip`](../../_downloads/03779dae39d11b141b20f09aaaa72f1b/multi_stream_encoding.zip)

[Gallery generated by Sphinx-Gallery](https://sphinx-gallery.github.io)