# TorchCodec

TorchCodec is a Python package with a goal to provide useful and fast APIs to
decode video frames to Pytorch Tensors. Currently decoding happens on the CPU
and we plan to add GPU acceleration.

## Installing TorchCodec

```bash
pip install torchcodec
```

## Using TorchCodec

A simple example showing how to decode video frames from a video file:

```python
# A simple example using TorchCodec
from torchcodec.decoders import SimpleVideoDecoder
video = SimpleVideoDecoder("/path/to/video.mp4")
frame_count = len(video)
# Returned frames are Pytorch Tensors with shape HWC.
first_frame = video[0]
last_frame = video[-1]
frame_visible_at_2_seconds = video.get_frame_displayed_at(2)
```

For more examples look at the `examples` directory.

## Installing from source

Use a conda or virtual environment.

Install torch nightly, e.g.:

```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
```

For more options, e.g. if you need CUDA or prefer using `conda`, refer to the
[official page](https://pytorch.org/get-started/locally/).

```bash
BUILD_AGAINST_ALL_FFMPEG_FROM_S3=1 pip install -e ".[dev]" --no-build-isolation -vvv
```

If you prefer building against an installed version of FFmpeg you can ommit the
`BUILD_AGAINST_ALL_FFMPEG_FROM_S3=1` part. Make sure `pkg-config` is installed
and able to find your FFmpeg installation. The easiest way to do this is to
install both from conda:

```bash
conda install ffmpeg pkg-config -c conda-forge
BUILD_AGAINST_ALL_FFMPEG_FROM_S3=1
```

## Building the docs

First install from source, then install the doc dependencies:

```bash
cd docs
pip install -r requirements.txt
```

Then, still from within the `docs` directory:

```bash
make html
```

The built docs will be in `build/html`. Open in your browser to view them.

To avoid building the examples (which execute python code and can take time) you
can use `make html-noplot`. To build a subset of specific examples instead of
all of them, you can use a regex like
`EXAMPLES_PATTERN="plot_the_best_example*" make html`.

Run `make clean` from time to time if you encounter issues.
