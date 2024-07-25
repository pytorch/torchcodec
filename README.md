<!-- TODO_BEFORE_RELEASE Add obvious link to docs and potentially a tag as
well-->

# TorchCodec

TorchCodec is a Python package with a goal to provide useful and fast APIs to
decode video frames to PyTorch Tensors.

⚠️ TorchCodec is still in early development stage and we are actively seeking
feedback. If you have any suggestions or issues, please let us know by opening
an issue!
<!-- TODO_UPDATE_LINK add link to issue tracker -->

## Using TorchCodec

Here's a condensed summary of what you can do with TorchCodec. For a more
detailed example, [check out our
documentation](https://pytorch.org/torchcodec/main/generated_examples/)!

```python
from torchcodec.decoders import SimpleVideoDecoder

# Note: If you don't have a video file at hand, you can generate one using the
# snippet below
decoder = SimpleVideoDecoder("/tmp/output_video.mp4")

decoder.metadata
# VideoStreamMetadata:  (Truncated output)
#   num_frames: 250
#   duration_seconds: 10.0
#   bit_rate: 31315.0
#   codec: h264
#   average_fps: 25.0

# Indexing API
decoder[0]  # Tensor of shape [C, H, W]
decoder[0 : -1 : 20]  # Stacked Tensor of shape [N, C, H, W]

# Indexing, with PTS and duration info
decoder.get_frame_at(len(decoder) - 1)
# Frame:
#   data (shape): torch.Size([3, 400, 640])
#   pts_seconds: 9.960000038146973
#   duration_seconds: 0.03999999910593033

decoder.get_frames_at(start=10, stop=30, step=5)
# FrameBatch:
#   data (shape): torch.Size([4, 3, 400, 640])
#   pts_seconds: tensor([0.4000, 0.6000, 0.8000, 1.0000])
#   duration_seconds: tensor([0.0400, 0.0400, 0.0400, 0.0400])

# Time-based indexing with PTS and duration info
decoder.get_frame_displayed_at(pts_seconds=2)
# Frame:
#   data (shape): torch.Size([3, 400, 640])
#   pts_seconds: 2.0
#   duration_seconds: 0.03999999910593033
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

First install the latest stable version of PyTorch following the [official
instructions](https://pytorch.org/get-started/locally/).

Then:

```bash
pip install torchcodec
```
You will also need FFmpeg installed on your system, and TorchCodec decoding
capabilities are determined by your underlying FFmpeg installation. There are
different options to install FFmpeg e.g.:

```bash

    conda install ffmpeg
    # or
    conda install ffmpeg -c conda-forge
```

Your Linux distribution probably comes with FFmpeg pre-installed as well.
TorchCodec supports all major FFmpeg version in [4, 7].


## Planned future work

We are actively working on the following features:

<!-- TODO_UPDATE_LINK link to relevant issues-->
- MacOS support (currently, only Linux is supported)
- GPU decoding
- Audio decoding

Let us know if you have any feature requests by opening an issue!

## Contributing

We welcome contributions to TorchCodec! Please see our [contributing
guide](CONTRIBUTING.md) for more details.

## License

TorchCodec is released under the [BSD 3 license](./LICENSE).
