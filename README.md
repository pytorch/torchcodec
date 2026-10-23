[**Installation**](#installing-torchcodec) | [**Documentation**](https://meta-pytorch.org/torchcodec) | [**Contributing**](CONTRIBUTING.md) | [**License**](#license)

<p align="center">
  <img src="https://raw.githubusercontent.com/meta-pytorch/torchcodec/main/docs/source/_static/img/torchcodec_horizontal_logo.jpg" alt="TorchCodec" width="500">
</p>

TorchCodec is a PyTorch-native library for decoding and encoding media: videos,
audio, and images, on CPU and CUDA GPU. It aims to be fast, easy to
use, and well integrated into the PyTorch ecosystem. If you want to use PyTorch
to train ML models on videos, audio, or images, or run inference, TorchCodec is
how you turn these into tensors, and back.

We achieve these capabilities through:

* Pythonic APIs that mirror Python and PyTorch conventions.
* Relying on [FFmpeg](https://www.ffmpeg.org/) to do the video and audio
  decoding and encoding. TorchCodec uses the version of FFmpeg you already have
  installed. FFmpeg is a mature library with broad coverage available on most
  systems. It is, however, not easy to use. TorchCodec abstracts FFmpeg's
  complexity to ensure it is used correctly and efficiently. (FFmpeg is
  optional, and the image decoders and encoders don't need it: see [Installing
  TorchCodec](#installing-torchcodec).)
* Returning data as PyTorch tensors, ready to be fed into PyTorch transforms
  or used directly to train models.

## Usage Examples

Below are some examples of what you can do with TorchCodec. For more detailed
examples and more use-cases, [check out our
documentation](https://meta-pytorch.org/torchcodec/stable/generated_examples/)!

#### Video Decoding

```python
from torchcodec.decoders import VideoDecoder

device = "cpu"  # or e.g. "cuda" !
decoder = VideoDecoder("path/to/video.mp4", device=device)

decoder.metadata
# VideoStreamMetadata:
#   num_frames: 250
#   duration_seconds: 10.0
#   bit_rate: 31315.0
#   codec: h264
#   average_fps: 25.0
#   ... (truncated output)

# Simple Indexing API
decoder[0]  # uint8 tensor of shape [C, H, W]
decoder[0 : -1 : 20]  # uint8 stacked tensor of shape [N, C, H, W]

# Indexing, with PTS and duration info:
decoder.get_frames_at(indices=[2, 100])
# FrameBatch:
#   data (shape): torch.Size([2, 3, 270, 480])
#   pts_seconds: tensor([0.0667, 3.3367], dtype=torch.float64)
#   duration_seconds: tensor([0.0334, 0.0334], dtype=torch.float64)

# Time-based indexing with PTS and duration info
decoder.get_frames_played_at(seconds=[0.5, 10.4])
# FrameBatch:
#   data (shape): torch.Size([2, 3, 270, 480])
#   pts_seconds: tensor([ 0.4671, 10.3770], dtype=torch.float64)
#   duration_seconds: tensor([0.0334, 0.0334], dtype=torch.float64)
```

You can use the following snippet to generate a video with FFmpeg and try out
the `VideoDecoder`:

```bash
ffmpeg -f lavfi -i testsrc2=size=640x400:duration=10:rate=25 /tmp/output_video.mp4
```

#### Video and Audio Encoding

```python
from torchcodec.encoders import Encoder

encoder = Encoder()
video_stream = encoder.add_video(
    height=height, width=width, frame_rate=frame_rate,
)
audio_stream = encoder.add_audio(
    sample_rate=sample_rate, num_channels=num_channels,
)
with encoder.open_file("output.mp4"):
    video_stream.add_frames(frames_batch_0)
    audio_stream.add_samples(samples_batch_0)
    video_stream.add_frames(frames_batch_1)
    audio_stream.add_samples(samples_batch_1)
    # ...
```

#### Image Decoding and Encoding

```python
from torchcodec.decoders import decode_image, decode_jpeg
from torchcodec.encoders import JpegEncoder

# JPEG, PNG, WebP, GIF, AVIF and HEIC, with the format detected automatically.
image = decode_image("path/to/image.jpg")  # uint8 tensor of shape [C, H, W]

# Or use the format-specific decoders, e.g. to decode JPEGs on GPU:
image = decode_jpeg("path/to/image.jpg", device="cuda")

# JPEG and PNG encoding. JPEGEncoder also supports CUDA encoding!
JpegEncoder(image).to_file("output.jpg")  # also .to_tensor() and .to_file_like()
```

## Installing TorchCodec

1. Install FFmpeg, if it's not already installed. TorchCodec supports all major
   FFmpeg versions in [4, 9]. Linux distributions usually come with FFmpeg
   pre-installed. You'll need FFmpeg that comes with separate shared libraries.
   This is especially relevant for Windows users: these are usually called the
   "shared" releases.

   If FFmpeg is not already installed, or you need a more recent version, an
   easy way to install it is to use `conda`:

   ```bash
   conda install "ffmpeg"
   # or
   conda install "ffmpeg" -c conda-forge
   ```

   > **Note:** FFmpeg is an *optional* dependency. It is needed for video
   > and audio decoding and encoding (`VideoDecoder`, `AudioDecoder`,
   > `VideoEncoder`, `AudioEncoder`, etc.). The image decoders and encoders
   > (`decode_image`, `decode_jpeg`, `JpegEncoder`, `PngEncoder`, etc.)
   > do **not** require FFmpeg, so if you only need images you can skip
   > this step.

2. Install PyTorch and TorchCodec:

   ```bash
   pip install torch torchcodec
   ```

   That's it! On Linux x86 and aarch64, this will install CUDA-enabled wheels by
   default (matching the default behavior of `pip install torch`). These wheels
   should *still* work even if you do not have a GPU on your machine. On macOS
   and Windows this will install CPU-only wheels. CPU wheels are available for
   Linux (x86_64 and aarch64), macOS, and Windows.

### CUDA support

On CUDA GPUs, TorchCodec supports decoding and encoding of videos and jpeg
images. CUDA-enabled wheels are installed by default on Linux. For Windows,
you'll need to pass `--index-url` as described below.


For video, make sure you have a GPU with NVDEC and NVENC hardware that supports
the formats you want. Refer to Nvidia's GPU support matrix
[here](https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new).


To select a specific CUDA Toolkit version, use `--index-url`. Make sure to
install the corresponding PyTorch version as well (refer to the
[official instructions](https://pytorch.org/get-started/locally/)):

```bash
# This corresponds to CUDA Toolkit version 13.0.
pip install torch torchcodec --index-url=https://download.pytorch.org/whl/cu130
```

Make sure your FFmpeg has NVDEC and NVENC support:

```bash
ffmpeg -decoders | grep -i nvidia
# This should show a line like this:
# V..... h264_cuvid           Nvidia CUVID H264 decoder (codec h264)

ffmpeg -encoders | grep -i nvidia
# This should show a line like this:
# V....D h264_nvenc           NVIDIA NVENC H.264 encoder (codec h264)
```

To check that FFmpeg libraries work with NVDEC correctly you can decode a
generated test video:

```bash
ffmpeg -hwaccel cuda -hwaccel_output_format cuda -f lavfi -i testsrc2=duration=1 -f null -
```

### CPU-only installation

To install CPU-only wheels explicitly (e.g. on Linux where CUDA wheels are the
default):

```bash
pip install torchcodec --index-url=https://download.pytorch.org/whl/cpu
```

### XPU support

Intel GPUs (XPU) support requires a stand-alone plugin for TorchCodec:

```bash
pip install torchcodec-xpu --extra-index-url=https://download.pytorch.org/whl/xpu
```

For any XPU-related support, please refer to
https://github.com/intel/torchlib-xpu.


### Compatibility with `torch` versions

The following table indicates the compatibility between versions of
`torchcodec`, `torch` and Python.

| `torchcodec`       | `torch`            | Python              |
| ------------------ | ------------------ | ------------------- |
| `main` / `nightly` | `main` / `nightly` | `>=3.10`, `<=3.14`   |
| `0.17`             | `>=2.11`             | `>=3.10`, `<=3.14`   |
| `0.16`             | `>=2.11`             | `>=3.10`, `<=3.14`   |
| `0.15`             | `>=2.11`             | `>=3.10`, `<=3.14`   |
| `0.14`             | `>=2.11`             | `>=3.10`, `<=3.14`   |

<details>
    <summary>older versions</summary>

| `torchcodec`       | `torch`            | Python              |
| ------------------ | ------------------ | ------------------- |
| `0.13`             | `>=2.11`             | `>=3.10`, `<=3.14`   |
| `0.12`             | `>=2.11`             | `>=3.10`, `<=3.14`   |
| `0.11`             | `2.11`             | `>=3.10`, `<=3.14`   |
| `0.10`             | `2.10`             | `>=3.10`, `<=3.14`   |
| `0.9`              | `2.9`              | `>=3.10`, `<=3.14`   |
| `0.8`              | `2.9`              | `>=3.10`, `<=3.13`   |
| `0.7`              | `2.8`              | `>=3.9`, `<=3.13`   |
| `0.6`              | `2.8`              | `>=3.9`, `<=3.13`   |
| `0.5`              | `2.7`              | `>=3.9`, `<=3.13`   |
| `0.4`              | `2.7`              | `>=3.9`, `<=3.13`   |
| `0.3`              | `2.7`              | `>=3.9`, `<=3.13`   |
| `0.2`              | `2.6`              | `>=3.9`, `<=3.13`   |
| `0.1`              | `2.5`              | `>=3.9`, `<=3.12`   |
| `0.0.3`            | `2.4`              | `>=3.8`, `<=3.12`   |

</details>


## Contributing

We welcome contributions to TorchCodec! Please see our [contributing
guide](CONTRIBUTING.md) for more details.

## License

TorchCodec is released under the [BSD 3 license](./LICENSE).

However, TorchCodec may be used with code not written by Meta which may be
distributed under different licenses.

For example, if you build TorchCodec with ENABLE_CUDA=1 or use the CUDA-enabled
release of torchcodec, please review CUDA's license here:
[Nvidia licenses](https://docs.nvidia.com/cuda/eula/index.html).
