# Contributing to TorchCodec

You can contribute to this project by writing code, fixing issues or simply by
using the library and reporting your feedback.

## Development installation

### Dependencies

Start by installing the **nightly** build of PyTorch following the
[official instructions](https://pytorch.org/get-started/locally/). Note that the
official instructions may ask you to install TorchCodec itself. If you are doing
development on TorchCodec, you should not install prebuilt TorchCodec packages.

To build, run and test locally you will need the following packages
(dependencies):

1. C++ compiler+linker (g++) and C++ runtime library (libstdc++). This is
   typically available on a baseline Linux installation already.
1. python
1. cmake
1. pkg-config
1. libtorch (this is part of Pytorch)
1. ffmpeg
1. pytest (for testing)

You can install these using your favorite package manager. Example, for conda
use:

```bash
# After installing the nightly pytorch version run this:
conda install cmake pkg-config ffmpeg pytest -c conda-forge
```

### Clone and build torchcodec

```bash
git clone https://github.com/pytorch-labs/torchcodec.git
cd torchcodec

pip install -e ".[dev]" --no-build-isolation -vv

# This should decode the sample video and generate a png file that is a frame in the video.
python test/decoders/manual_smoke_test.py
```

### Running unit tests

To run python tests run:

```bash
pytest test -vvv
```

To run C++ tests run:

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=1 -DCMAKE_PREFIX_PATH=$(python3 -c 'import torch;print(torch.utils.cmake_prefix_path)) ..
cmake --build . -- VERBOSE=1
ctest --rerun-failed --output-on-failure
```

## Development Process

TBD

### Code formatting and typing

TODO: instructions for pre-commit and mypy

## License

By contributing to TorchCodec, you agree that your contributions will be
licensed under the LICENSE file in the root directory of this source tree.

Contributors are also required to
[sign our Contributor License Agreement](https://code.facebook.com/cla).
