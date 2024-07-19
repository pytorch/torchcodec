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

<!-- TODO_UPDATE_LINK add link to docs --> 
For more detailed examples, check out our docs!

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
