# Decoding with custom frame mappings

In this example, we will describe the `custom_frame_mappings` parameter of the
[`VideoDecoder`](../../generated/torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder) class.
This parameter allows you to provide pre-computed frame mapping information to
speed up [`VideoDecoder`](../../generated/torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder) instantiation, while
maintaining the frame seeking accuracy of `seek_mode="exact"`.

This makes it ideal for workflows where:

> 1. Frame accuracy is critical, so [approximate mode](approximate_mode.html) cannot be used
> 2. Videos can be preprocessed once and then decoded many times

First, some boilerplate: we'll download a short video from the web, and
use ffmpeg to create a longer version by repeating it multiple times. We'll end up
with two videos: a short one of approximately 14 seconds and a long one of about 12 minutes.
You can ignore this part and skip below to Creating custom frame mappings with ffprobe.

```
import tempfile
from pathlib import Path
import subprocess
import requests

# Video source: https://www.pexels.com/video/dog-eating-854132/
# License: CC0. Author: Coverr.
url = "https://videos.pexels.com/video-files/854132/854132-sd_640_360_25fps.mp4"
response = requests.get(url, headers={"User-Agent": ""})
if response.status_code != 200:
 raise RuntimeError(f"Failed to download video. {response.status_code = }.")

temp_dir = tempfile.mkdtemp()
short_video_path = Path(temp_dir) / "short_video.mp4"
with open(short_video_path, 'wb') as f:
 for chunk in response.iter_content():
 f.write(chunk)

long_video_path = Path(temp_dir) / "long_video.mp4"
ffmpeg_command = [
 "ffmpeg",
 "-stream_loop", "50", # repeat video 50 times to get a ~12 min video
 "-i", f"{short_video_path}",
 "-c", "copy",
 f"{long_video_path}"
]
subprocess.run(ffmpeg_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

from torchcodec.decoders import VideoDecoder
print(f"Short video duration: {VideoDecoder(short_video_path).metadata.duration_seconds} seconds")
print(f"Long video duration: {VideoDecoder(long_video_path).metadata.duration_seconds / 60} minutes")
```

```
Short video duration: 13.8 seconds
Long video duration: 11.729999999999999 minutes
```

## Creating custom frame mappings with ffprobe

To generate JSON files containing the required video metadata, we recommend using ffprobe.
The following frame metadata fields are needed
(the `pkt_` prefix is needed for older versions of FFmpeg):

- `pts` / `pkt_pts`: Presentation timestamps for each frame
- `duration` / `pkt_duration`: Duration of each frame
- `key_frame`: Boolean indicating which frames are key frames

```
from pathlib import Path
import subprocess
import tempfile
from time import perf_counter_ns
import json

# Lets define a simple function to run ffprobe on a video's first stream index, then writes the results in output_json_path.
def generate_frame_mappings(video_path, output_json_path, stream_index):
 ffprobe_cmd = [
 "ffprobe",
 "-i", f"{video_path}",
 "-select_streams", f"{stream_index}",
 "-show_frames",
 "-show_entries",
 "frame=pts,duration,key_frame",
 "-of", "json",
 ]
 print(f"Running ffprobe:\n{' '.join(ffprobe_cmd)}\n")
 ffprobe_result = subprocess.run(ffprobe_cmd, check=True, capture_output=True, text=True)
 with open(output_json_path, "w") as f:
 f.write(ffprobe_result.stdout)

stream_index = 0
long_json_path = Path(temp_dir) / "long_custom_frame_mappings.json"
short_json_path = Path(temp_dir) / "short_custom_frame_mappings.json"

generate_frame_mappings(long_video_path, long_json_path, stream_index)
generate_frame_mappings(short_video_path, short_json_path, stream_index)
with open(short_json_path) as f:
 sample_data = json.loads(f.read())
print("Sample of fields in custom frame mappings:")
for frame in sample_data["frames"][:3]:
 print(f"{frame['key_frame'] = }, {frame['pts'] = }, {frame['duration'] = }")
```

```
Running ffprobe:
ffprobe -i /tmp/tmps8syvoy1/long_video.mp4 -select_streams 0 -show_frames -show_entries frame=pts,duration,key_frame -of json

Running ffprobe:
ffprobe -i /tmp/tmps8syvoy1/short_video.mp4 -select_streams 0 -show_frames -show_entries frame=pts,duration,key_frame -of json

Sample of fields in custom frame mappings:
frame['key_frame'] = 1, frame['pts'] = 0, frame['duration'] = 1
frame['key_frame'] = 0, frame['pts'] = 1, frame['duration'] = 1
frame['key_frame'] = 0, frame['pts'] = 2, frame['duration'] = 1
```

## Performance: `VideoDecoder` creation

Custom frame mappings affect the **creation** of a [`VideoDecoder`](../../generated/torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder)
object. As video length or resolution increases, the performance gain compared to exact mode increases.

```
import torch

# Here, we define a benchmarking function, with the option to seek to the start of a file_like.
def bench(f, file_like=False, average_over=50, warmup=2, **f_kwargs):
 for _ in range(warmup):
 f(**f_kwargs)
 if file_like:
 f_kwargs["custom_frame_mappings"].seek(0)

 times = []
 for _ in range(average_over):
 start = perf_counter_ns()
 f(**f_kwargs)
 end = perf_counter_ns()
 times.append(end - start)
 if file_like:
 f_kwargs["custom_frame_mappings"].seek(0)

 times = torch.tensor(times) * 1e-6 # ns to ms
 std = times.std().item()
 med = times.median().item()
 print(f"{med = :.2f}ms +- {std:.2f}")

for video_path, json_path in ((short_video_path, short_json_path), (long_video_path, long_json_path)):
 print(f"\nRunning benchmarks on {Path(video_path).name}")

 print("Creating a VideoDecoder object with custom_frame_mappings:")
 with open(json_path, "r") as f:
 bench(VideoDecoder, file_like=True, source=video_path, stream_index=stream_index, custom_frame_mappings=f)

 # Compare against exact seek_mode
 print("Creating a VideoDecoder object with seek_mode='exact':")
 bench(VideoDecoder, source=video_path, stream_index=stream_index, seek_mode="exact")
```

```
Running benchmarks on short_video.mp4
Creating a VideoDecoder object with custom_frame_mappings:
med = 6.19ms +- 0.03
Creating a VideoDecoder object with seek_mode='exact':
med = 6.45ms +- 0.02

Running benchmarks on long_video.mp4
Creating a VideoDecoder object with custom_frame_mappings:
med = 27.42ms +- 0.14
Creating a VideoDecoder object with seek_mode='exact':
med = 42.93ms +- 0.36
```

## Performance: Frame decoding with custom frame mappings

Although using `custom_frame_mappings` only impacts the initialization speed of
[`VideoDecoder`](../../generated/torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder), decoding workflows
involve creating a [`VideoDecoder`](../../generated/torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder) instance,
so the performance benefits are realized.

```
def decode_frames(video_path, seek_mode="exact", custom_frame_mappings=None):
 decoder = VideoDecoder(
 source=video_path,
 seek_mode=seek_mode,
 custom_frame_mappings=custom_frame_mappings
 )
 decoder.get_frames_in_range(start=0, stop=10)

for video_path, json_path in ((short_video_path, short_json_path), (long_video_path, long_json_path)):
 print(f"\nRunning benchmarks on {Path(video_path).name}")
 print("Decoding frames with custom_frame_mappings:")
 with open(json_path, "r") as f:
 bench(decode_frames, file_like=True, video_path=video_path, custom_frame_mappings=f)

 print("Decoding frames with seek_mode='exact':")
 bench(decode_frames, video_path=video_path, seek_mode="exact")
```

```
Running benchmarks on short_video.mp4
Decoding frames with custom_frame_mappings:
med = 18.31ms +- 0.03
Decoding frames with seek_mode='exact':
med = 18.56ms +- 0.05

Running benchmarks on long_video.mp4
Decoding frames with custom_frame_mappings:
med = 39.96ms +- 0.21
Decoding frames with seek_mode='exact':
med = 56.12ms +- 0.47
```

## Accuracy: Metadata and frame retrieval

In addition to the instantiation speed up compared to `seek_mode="exact"`, using custom frame mappings
also retains the benefit of exact metadata and frame seeking.

```
print("Metadata of short video with custom_frame_mappings:")
with open(short_json_path, "r") as f:
 print(VideoDecoder(short_video_path, custom_frame_mappings=f).metadata)
print("Metadata of short video with seek_mode='exact':")
print(VideoDecoder(short_video_path, seek_mode="exact").metadata)

with open(short_json_path, "r") as f:
 custom_frame_mappings_decoder = VideoDecoder(short_video_path, custom_frame_mappings=f)
exact_decoder = VideoDecoder(short_video_path, seek_mode="exact")
for i in range(len(exact_decoder)):
 torch.testing.assert_close(
 exact_decoder.get_frame_at(i).data,
 custom_frame_mappings_decoder.get_frame_at(i).data,
 atol=0, rtol=0,
 )
print("Frame seeking is the same for this video!")
```

```
Metadata of short video with custom_frame_mappings:
VideoStreamMetadata:
 duration_seconds_from_header: 13.8
 begin_stream_seconds_from_header: 0
 bit_rate: 505790
 codec: h264
 stream_index: 0
 width: 640
 height: 360
 num_frames_from_header: 345
 average_fps_from_header: 25
 pixel_aspect_ratio: 1
 rotation: None
 color_primaries: smpte170m
 color_space: smpte170m
 color_transfer_characteristic: smpte170m
 pixel_format: yuv420p
 begin_stream_seconds_from_content: None
 end_stream_seconds_from_content: None
 num_frames_from_content: None
 duration_seconds: 13.8
 begin_stream_seconds: 0
 end_stream_seconds: 13.8
 num_frames: 345
 average_fps: 25

Metadata of short video with seek_mode='exact':
VideoStreamMetadata:
 duration_seconds_from_header: 13.8
 begin_stream_seconds_from_header: 0
 bit_rate: 505790
 codec: h264
 stream_index: 0
 width: 640
 height: 360
 num_frames_from_header: 345
 average_fps_from_header: 25
 pixel_aspect_ratio: 1
 rotation: None
 color_primaries: smpte170m
 color_space: smpte170m
 color_transfer_characteristic: smpte170m
 pixel_format: yuv420p
 begin_stream_seconds_from_content: 0
 end_stream_seconds_from_content: 13.8
 num_frames_from_content: 345
 duration_seconds: 13.8
 begin_stream_seconds: 0
 end_stream_seconds: 13.8
 num_frames: 345
 average_fps: 25

Frame seeking is the same for this video!
```

## How do custom_frame_mappings help?

Custom frame mappings contain the same frame index information
that would normally be computed during the [scan](../../glossary.html#term-scan) operation in exact mode.
By providing this information to the [`VideoDecoder`](../../generated/torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder)
as a JSON, it eliminates the need for the expensive scan while preserving the
accuracy benefits.

## Which mode should I use?

- For fastest decoding when speed is more important than exact seeking accuracy,
"approximate" mode is recommended.
- For exact frame seeking, custom frame mappings will benefit workflows where the
same videos are decoded repeatedly, and some preprocessing work can be done.
- For exact frame seeking without preprocessing, use "exact" mode.

**Total running time of the script:** (0 minutes 23.826 seconds)

[`Download Jupyter notebook: custom_frame_mappings.ipynb`](../../_downloads/515bb6477ed8fb530cccbe1b67ef6f0c/custom_frame_mappings.ipynb)

[`Download Python source code: custom_frame_mappings.py`](../../_downloads/3a6f108995251f5767dcacb2c6ed4656/custom_frame_mappings.py)

[`Download zipped: custom_frame_mappings.zip`](../../_downloads/e21d3326a4a2f21f4fb841be7e71394f/custom_frame_mappings.zip)

[Gallery generated by Sphinx-Gallery](https://sphinx-gallery.github.io)