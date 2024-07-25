<!-- TODO_BEFORE_RELEASE Add obvious link to docs and potentially a tag as
well-->

# TorchCodec

TorchCodec is a Python package with a goal to provide useful and fast APIs to
decode video frames to PyTorch Tensors.

⚠️ TorchCodec is still in early development stage and we are actively seeking
feedback. If you have any suggestions or issues, please let us know by [opening
an issue](https://github.com/pytorch/torchcodec/issues/new/choose)!

## Using TorchCodec

<!-- TODO BEFORE_RELEASE: polish this example -->
```python
from torchcodec.decoders import SimpleVideoDecoder

video = SimpleVideoDecoder("/path/to/video.mp4")

# Indexing API:
first_frame = video[0]
last_frame = video[-1]

# PTS API:
frame_visible_at_2_seconds = video.get_frame_displayed_at(2)
```

For more detailed examples, [check out our docs](https://pytorch.org/torchcodec/stable/index.html)!

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

- [MacOS support](https://github.com/pytorch/torchcodec/issues/111) (currently, only Linux is supported)
- [GPU decoding](https://github.com/pytorch/torchcodec/pull/58)
- [Audio decoding](https://github.com/pytorch/torchcodec/issues/85)

Let us know if you have any feature requests by [opening an
issue](https://github.com/pytorch/torchcodec/issues/new?assignees=&labels=&projects=&template=feature-request.yml)!

## Contributing

We welcome contributions to TorchCodec! Please see our [contributing
guide](CONTRIBUTING.md) for more details.

## License

TorchCodec is released under the [BSD 3 license](./LICENSE).
