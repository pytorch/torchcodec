Installation Instructions
=========================

We'll be providing wheels in the coming days so that you can just install
torchcodec using ``pip``. For now, you can just build from source. You will need
the following dependencies:

- A C++ compiler+linker. This is typically available on a baseline Linux
  installation already.
- cmake
- pkg-config
- FFmpeg
- PyTorch nightly

Start by installing the **nightly** build of PyTorch following the
`official instructions <https://pytorch.org/get-started/locally/>_.

Then, the easiest way to install the rest of the dependencies is to run:

    conda install cmake pkg-config ffmpeg -c conda-forge

To clone and install the repo, run:

    git clone git@github.com:pytorch/torchcodec.git
    # Or, using https instead of ssh: git clone https://github.com/pytorch/torchcodec.git
    cd torchcodec

    pip install -e ".[dev]" --no-build-isolation -vv

TorchCodec supports all major FFmpeg version in [4, 7].
