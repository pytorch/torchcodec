# Streaming data through file-like support

In this example, we will show how to decode streaming data. That is, when files
do not reside locally, we will show how to only download the data segments that
are needed to decode the frames you care about. We accomplish this capability
with Python
[file-like objects](https://docs.python.org/3/glossary.html#term-file-like-object).
Our example uses a video file, so we use the [`VideoDecoder`](../../generated/torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder)
class to decode it. But all of the lessons here also apply to audio files and the
[`AudioDecoder`](../../generated/torchcodec.decoders.AudioDecoder.html#torchcodec.decoders.AudioDecoder) class as well.

First, a bit of boilerplate. We define two functions: one to download content
from a given URL, and another to time the execution of a given function.

```
import torch
import requests
from time import perf_counter_ns

def get_url_content(url):
 response = requests.get(url, headers={"User-Agent": ""})
 if response.status_code != 200:
 raise RuntimeError(f"Failed to download video. {response.status_code = }.")
 return response.content

def bench(f, average_over=10, warmup=2):
 for _ in range(warmup):
 f()

 times = []
 for _ in range(average_over):
 start = perf_counter_ns()
 f()
 end = perf_counter_ns()
 times.append(end - start)

 times = torch.tensor(times) * 1e-6 # ns to ms
 std = times.std().item()
 med = times.median().item()
 print(f"{med = :.2f}ms +- {std:.2f}")
```

## Performance: downloading first vs. streaming

We are going to investigate the cost of having to download an entire video
before decoding any frames versus being able to stream the video's data
while decoding. To demonsrate an extreme case, we're going to always decode
just the first frame of the video, while we vary how we get that video's
data.

The video we're going to use in this tutorial is publicly available on the
internet. We perform an initial download of it so that we can understand
its size and content:

```
from torchcodec.decoders import VideoDecoder

nasa_url = "https://download.pytorch.org/torchaudio/tutorial-assets/stream-api/NASAs_Most_Scientifically_Complex_Space_Observatory_Requires_Precision-MP4.mp4"

pre_downloaded_raw_video_bytes = get_url_content(nasa_url)
decoder = VideoDecoder(pre_downloaded_raw_video_bytes)

print(f"Video size in MB: {len(pre_downloaded_raw_video_bytes) // 1024 // 1024}")
print(decoder.metadata)
```

```
Video size in MB: 253
VideoStreamMetadata:
 duration_seconds_from_header: 206.03916666666666
 begin_stream_seconds_from_header: 0
 bit_rate: 9958354
 codec: h264
 stream_index: 0
 width: 1920
 height: 1080
 num_frames_from_header: 6175
 average_fps_from_header: 29.97002997002997
 pixel_aspect_ratio: 1
 rotation: None
 color_primaries: bt709
 color_space: bt709
 color_transfer_characteristic: bt709
 pixel_format: yuv420p
 begin_stream_seconds_from_content: 0
 end_stream_seconds_from_content: 206.03916666666666
 num_frames_from_content: 6175
 duration_seconds: 206.03916666666666
 begin_stream_seconds: 0
 end_stream_seconds: 206.03916666666666
 num_frames: 6175
 average_fps: 29.97002997002997
```

We can see that the video is about 253 MB, has the resolution 1920x1080, is
about 30 frames per second and is almost 3 and a half minutes long. As we
only want to decode the first frame, we would clearly benefit from not having
to download the entire video!

Let's first test three scenarios:

> 1. Decode from the *existing* video we just downloaded. This is our baseline
> performance, as we've reduced the downloading cost to 0.
> 2. Download the entire video before decoding. This is the worst case
> that we want to avoid.
> 3. Provde the URL directly to the [`VideoDecoder`](../../generated/torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder) class, which will pass
> the URL on to FFmpeg. Then FFmpeg will decide how much of the video to
> download before decoding.

Note that in our scenarios, we are always setting the `seek_mode` parameter of
the [`VideoDecoder`](../../generated/torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder) class to `"approximate"`. We do
this to avoid scanning the entire video during initialization, which would
require downloading the entire video even if we only want to decode the first
frame. See [Exact vs Approximate seek mode: Performance and accuracy comparison](approximate_mode.html#sphx-glr-generated-examples-decoding-approximate-mode-py) for more.

```
def decode_from_existing_download():
 decoder = VideoDecoder(
 source=pre_downloaded_raw_video_bytes,
 seek_mode="approximate",
 )
 return decoder[0]

def download_before_decode():
 raw_video_bytes = get_url_content(nasa_url)
 decoder = VideoDecoder(
 source=raw_video_bytes,
 seek_mode="approximate",
 )
 return decoder[0]

def direct_url_to_ffmpeg():
 decoder = VideoDecoder(
 source=nasa_url,
 seek_mode="approximate",
 )
 return decoder[0]

print("Decode from existing download:")
bench(decode_from_existing_download)
print()

print("Download before decode:")
bench(download_before_decode)
print()

print("Direct url to FFmpeg:")
bench(direct_url_to_ffmpeg)
```

```
Decode from existing download:
med = 194.88ms +- 0.98

Download before decode:
med = 1234.19ms +- 127.76

Direct url to FFmpeg:
med = 268.40ms +- 13.24
```

Decoding the already downloaded video is clearly the fastest. Having to
download the entire video each time we want to decode just the first frame
is many times slower than decoding an existing video. Providing a direct URL
is much better, but we're still probably downloading more than we need to.

We can do better, and the way how is to use a file-like object which
implements its own read and seek methods that only download data from a URL as
needed. Rather than implementing our own, we can use such objects from the
[fsspec](https://github.com/fsspec/filesystem_spec) module that provides
[Filesystem interfaces for Python](https://filesystem-spec.readthedocs.io/en/latest/?badge=latest).
Note that using these capabilities from the fsspec library also requires the
[aiohttp](https://docs.aiohttp.org/en/stable/) module. You can install both with
pip install fsspec aiohttp.

```
import fsspec

def stream_while_decode():
 # The `client_kwargs` are passed down to the aiohttp module's client
 # session; we need to indicate that we need to trust the environment
 # settings for proxy configuration. Depending on your environment, you may
 # not need this setting.
 with fsspec.open(nasa_url, client_kwargs={'trust_env': True}) as file_like:
 decoder = VideoDecoder(file_like, seek_mode="approximate")
 return decoder[0]

print("Stream while decode: ")
bench(stream_while_decode)
```

```
Stream while decode:
med = 214.20ms +- 4.55
```

Streaming the data through a file-like object is much faster than
downloading the video first. And not only is it also faster than
providing a direct URL, it's more general. [`VideoDecoder`](../../generated/torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder) supports
direct URLs because the underlying FFmpeg functions support them. But the
kinds of protocols supported are determined by what that version of FFmpeg
supports. A file-like object can adapt any kind of resource, including ones
that are specific to your own infrastructure and are unknown to FFmpeg.

## How it works

In Python, a [file-like object](https://docs.python.org/3/glossary.html#term-file-like-object)
is any object that exposes special methods for reading, writing and seeking.
While such methods are obviously file oriented, it's not required that
a file-like object is backed by an actual file. As far as Python is concerned,
if an object acts like a file, it's a file. This is a powerful concept, as
it enables libraries that read or write data to assume a file-like interface.
Other libraries that present novel resources can then be easily used by
providing a file-like wrapper for their resource.

For our case, we only need the read and seek methods for decoding. The exact
method signature needed is in the example below. Rather than wrap a novel
resource, we demonstrate this capability by wrapping an actual file while
counting how often each method is called.

```
from pathlib import Path
import tempfile

# Create a local file to interact with.
temp_dir = tempfile.mkdtemp()
nasa_video_path = Path(temp_dir) / "nasa_video.mp4"
with open(nasa_video_path, "wb") as f:
 f.write(pre_downloaded_raw_video_bytes)

# A file-like class that is backed by an actual file, but it intercepts reads
# and seeks to maintain counts.
class FileOpCounter:
 def __init__(self, file):
 self._file = file
 self.num_reads = 0
 self.num_seeks = 0

 def read(self, size: int) -> bytes:
 self.num_reads += 1
 return self._file.read(size)

 def seek(self, offset: int, whence: int) -> int:
 self.num_seeks += 1
 return self._file.seek(offset, whence)

# Let's now get a file-like object from our class defined above, providing it a
# reference to the file we created. We pass our file-like object to the decoder
# rather than the file itself.
file_op_counter = FileOpCounter(open(nasa_video_path, "rb"))
counter_decoder = VideoDecoder(file_op_counter, seek_mode="approximate")

print("Decoder initialization required "
 f"{file_op_counter.num_reads} reads and "
 f"{file_op_counter.num_seeks} seeks.")

init_reads = file_op_counter.num_reads
init_seeks = file_op_counter.num_seeks

first_frame = counter_decoder[0]

print("Decoding the first frame required "
 f"{file_op_counter.num_reads - init_reads} additional reads and "
 f"{file_op_counter.num_seeks - init_seeks} additional seeks.")
```

```
Decoder initialization required 9 reads and 11 seeks.
Decoding the first frame required 2 additional reads and 1 additional seeks.
```

While we defined a simple class primarily for demonstration, it's actually
useful for diagnosing how much reading and seeking are required for different
decoding operations. We've also introduced a mystery that we should answer:
why does *initializing* the decoder take more reads and seeks than decoding
the first frame? The answer is that in our decoder implementation, we're
actually calling a special
[FFmpeg function](https://ffmpeg.org/doxygen/6.1/group__lavf__decoding.html#gad42172e27cddafb81096939783b157bb)
that decodes the first few frames to return more robust metadata.

It's also worth noting that the Python file-like interface is only half of
the story. FFmpeg also has its own mechanism for directing reads and seeks
during decoding to user-define functions. The
[`VideoDecoder`](../../generated/torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder) object does the work of
connecting the Python methods you define to FFmpeg. All you have to do is
define your methods in Python, and we do the rest.

## Performance: local file path vs. local file-like object

Since we have a local file defined, let's do a bonus performance test. We now
have two means of providing a local file to [`VideoDecoder`](../../generated/torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder):

> 1. Through a *path*, where the [`VideoDecoder`](../../generated/torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder)
> object will then do the work of opening the local file at that path.
> 2. Through a *file-like object*, where you open the file yourself and provide
> the file-like object to [`VideoDecoder`](../../generated/torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder).

An obvious question is: which is faster? The code below tests that question.

```
def decode_from_existing_file_path():
 decoder = VideoDecoder(nasa_video_path, seek_mode="approximate")
 return decoder[0]

def decode_from_existing_open_file_object():
 with open(nasa_video_path, "rb") as file:
 decoder = VideoDecoder(file, seek_mode="approximate")
 return decoder[0]

print("Decode from existing file path:")
bench(decode_from_existing_file_path)
print()

print("Decode from existing open file object:")
bench(decode_from_existing_open_file_object)
```

```
Decode from existing file path:
med = 194.51ms +- 0.12

Decode from existing open file object:
med = 194.79ms +- 0.23
```

Thankfully, the answer is both means of decoding from a local file take about
the same amount of time. This result means that in your own code, you can use
whichever method is more convienient. What this result implies is that the
cost of actually reading and copying data dominates the cost of calling Python
methods while decoding.

Finally, let's clean up the local resources we created.

```
import shutil
shutil.rmtree(temp_dir)
```

**Total running time of the script:** (0 minutes 29.917 seconds)

[`Download Jupyter notebook: file_like.ipynb`](../../_downloads/f0caa5fce0f20da68647d009fb8dd8d0/file_like.ipynb)

[`Download Python source code: file_like.py`](../../_downloads/faa3df3e570ebdff28b0f7a14845477a/file_like.py)

[`Download zipped: file_like.zip`](../../_downloads/7f201e1a3c5c829f34ab3eada34ca73c/file_like.zip)

[Gallery generated by Sphinx-Gallery](https://sphinx-gallery.github.io)