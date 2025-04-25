[**Installation**](#installing-torchcodec) | [**Simple Example**](#using-torchcodec) | [**Detailed Example**](https://pytorch.org/torchcodec/stable/generated_examples/) | [**Documentation**](https://pytorch.org/torchcodec) | [**Contributing**](CONTRIBUTING.md) | [**License**](#license)

# TorchCodec

TorchCodec is a Python library for decoding video and audio data into PyTorch
tensors, on CPU and CUDA GPU. It aims to be fast, easy to use, and well
integrated into the PyTorch ecosystem. If you want to use PyTorch to train ML
models on videos and audio, TorchCodec is how you turn these into data.

We achieve these capabilities through:

* Pythonic APIs that mirror Python and PyTorch conventions.
* Relying on [FFmpeg](https://www.ffmpeg.org/) to do the decoding. TorchCodec
  uses the version of FFmpeg you already have installed. FFmpeg is a mature
  library with broad coverage available on most systems. It is, however, not
  easy to use. TorchCodec abstracts FFmpeg's complexity to ensure it is used
  correctly and efficiently.
* Returning data as PyTorch tensors, ready to be fed into PyTorch transforms
  or used directly to train models.

> [!NOTE]
> ⚠️ TorchCodec is still in development stage and some APIs may be updated
> in future versions, depending on user feedback.
> If you have any suggestions or issues, please let us know by
> [opening an issue](https://github.com/pytorch/torchcodec/issues/new/choose)!

## Using TorchCodec

Here's a condensed summary of what you can do with TorchCodec. For more detailed
examples, [check out our
documentation](https://pytorch.org/torchcodec/stable/generated_examples/)!

#### Decoding

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

#### Clip sampling

```python

from torchcodec.samplers import clips_at_regular_timestamps

clips_at_regular_timestamps(
    decoder,
    seconds_between_clip_starts=1.5,
    num_frames_per_clip=4,
    seconds_between_frames=0.1
)
# FrameBatch:
#   data (shape): torch.Size([9, 4, 3, 270, 480])
#   pts_seconds: tensor([[ 0.0000,  0.0667,  0.1668,  0.2669],
#         [ 1.4681,  1.5682,  1.6683,  1.7684],
#         [ 2.9696,  3.0697,  3.1698,  3.2699],
#         ... (truncated), dtype=torch.float64)
#   duration_seconds: tensor([[0.0334, 0.0334, 0.0334, 0.0334],
#         [0.0334, 0.0334, 0.0334, 0.0334],
#         [0.0334, 0.0334, 0.0334, 0.0334],
#         ... (truncated), dtype=torch.float64)
```

You can use the following snippet to generate a video with FFmpeg and tryout
TorchCodec:

```bash
fontfile=/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono-Bold.ttf
output_video_file=/tmp/output_video.mp4

ffmpeg -f lavfi -i \
    color=size=640x400:duration=10:rate=25:color=blue \
    -vf "drawtext=fontfile=${fontfile}:fontsize=30:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2:text='Frame %{frame_num}'" \
    ${output_video_file}
```

## Installing TorchCodec
### Installing CPU-only TorchCodec

1. Install the latest stable version of PyTorch following the
   [official instructions](https://pytorch.org/get-started/locally/). For other
   versions, refer to the table below for compatibility between versions of
   `torch` and `torchcodec`.

2. Install FFmpeg, if it's not already installed. Linux distributions usually
   come with FFmpeg pre-installed. TorchCodec supports all major FFmpeg versions
   in [4, 7].

   If FFmpeg is not already installed, or you need a more recent version, an
   easy way to install it is to use `conda`:

   ```bash
   conda install ffmpeg
   # or
   conda install ffmpeg -c conda-forge
   ```

3. Install TorchCodec:

   ```bash
   pip install torchcodec
   ```

The following table indicates the compatibility between versions of
`torchcodec`, `torch` and Python.

| `torchcodec`       | `torch`            | Python              |
| ------------------ | ------------------ | ------------------- |
| `main` / `nightly` | `main` / `nightly` | `>=3.9`, `<=3.13`   |
| `0.3`              | `2.7`              | `>=3.9`, `<=3.13`   |
| `0.2`              | `2.6`              | `>=3.9`, `<=3.13`   |
| `0.1`              | `2.5`              | `>=3.9`, `<=3.12`   |
| `0.0.3`            | `2.4`              | `>=3.8`, `<=3.12`   |

### Installing CUDA-enabled TorchCodec

First, make sure you have a GPU that has NVDEC hardware that can decode the
format you want. Refer to Nvidia's GPU support matrix for more details
[here](https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new).

1. Install Pytorch corresponding to your CUDA Toolkit using the
   [official instructions](https://pytorch.org/get-started/locally/). You'll
   need the `libnpp` and `libnvrtc` CUDA libraries, which are usually part of
   the CUDA Toolkit.

2. Install or compile FFmpeg with NVDEC support.
   TorchCodec with CUDA should work with FFmpeg versions in [4, 7].

   If FFmpeg is not already installed, or you need a more recent version, an
   easy way to install it is to use `conda`:

   ```bash
   conda install ffmpeg
   # or
   conda install ffmpeg -c conda-forge
   ```

   If you are building FFmpeg from source you can follow Nvidia's guide to
   configuring and installing FFmpeg with NVDEC support
   [here](https://docs.nvidia.com/video-technologies/video-codec-sdk/12.0/ffmpeg-with-nvidia-gpu/index.html).

   After installing FFmpeg make sure it has NVDEC support when you list the supported
   decoders:

   ```bash
   ffmpeg -decoders | grep -i nvidia
   # This should show a line like this:
   # V..... h264_cuvid           Nvidia CUVID H264 decoder (codec h264)
   ```

   To check that FFmpeg libraries work with NVDEC correctly you can decode a sample video:

   ```bash
   ffmpeg -hwaccel cuda -hwaccel_output_format cuda -i test/resources/nasa_13013.mp4 -f null -
   ```

3. Install TorchCodec by passing in an `--index-url` parameter that corresponds
   to your CUDA Toolkit version, example:

   ```bash
   # This corresponds to CUDA Toolkit version 12.6. It should be the same one
   # you used when you installed PyTorch (If you installed PyTorch with pip).
   pip install torchcodec --index-url=https://download.pytorch.org/whl/cu126
   ```

   Note that without passing in the `--index-url` parameter, `pip` installs
   the CPU-only version of TorchCodec.

## Benchmark Results

The following was generated by running [our benchmark script](./benchmarks/decoders/generate_readme_data.py) on a lightly loaded 22-core machine with an Nvidia A100 with
5 [NVDEC decoders](https://docs.nvidia.com/video-technologies/video-codec-sdk/12.1/nvdec-application-note/index.html#).

![benchmark_results](./benchmarks/decoders/benchmark_readme_chart.png)

The top row is a [Mandelbrot](https://ffmpeg.org/ffmpeg-filters.html#mandelbrot) video
generated from FFmpeg that has a resolution of 1280x720 at 60 fps and is 120 seconds long.
The bottom row is [promotional video from NASA](https://download.pytorch.org/torchaudio/tutorial-assets/stream-api/NASAs_Most_Scientifically_Complex_Space_Observatory_Requires_Precision-MP4_small.mp4)
that has a resolution of 960x540 at 29.7 fps and is 206 seconds long. Both videos were
encoded with libx264 and yuv420p pixel format. All decoders, except for TorchVision, used FFmpeg 6.1.2. TorchVision used FFmpeg 4.2.2.

For TorchCodec, the "approx" label means that it was using [approximate mode](https://pytorch.org/torchcodec/stable/generated_examples/approximate_mode.html)
for seeking.

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
