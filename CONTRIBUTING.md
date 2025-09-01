# Contributing to TorchCodec

You can contribute to this project by writing code, fixing issues or simply by
using the library and reporting your feedback.

Below are instructions to build TorchCodec from source, as well as the usual
contribution guidelines (code formatting, testing, etc). To submit a PR, please
follow the [official GitHub
guidelines](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork).

## Building TorchCodec from source

### Installing dependencies

The instructions below assume you are using a conda environment, but the steps
are easily adaptable to other kind of virtual environments. To build, run and
test locally you will need the following dependencies:

- A C++ compiler+linker. This is typically available on a baseline Linux
  installation already.
- cmake
- pkg-config
- pybind11
- FFmpeg
- PyTorch nightly

Start by installing the **nightly** build of PyTorch following the
[official instructions](https://pytorch.org/get-started/locally/).

Then, the easiest way to install the rest of the dependencies is to run:

```bash
conda install cmake pkg-config pybind11 "ffmpeg<8" -c conda-forge
```

### Clone and build

To clone and install the repo, run:

```bash
git clone git@github.com:pytorch/torchcodec.git
# Or, using https instead of ssh: git clone https://github.com/pytorch/torchcodec.git
cd torchcodec

# Optional, but recommended: define a persistent build directory which speeds-up
# subsequent builds.
export TORCHCODEC_CMAKE_BUILD_DIR="${PWD}/build"

pip install -e ".[dev]" --no-build-isolation -vv
# Or, for cuda support: ENABLE_CUDA=1 pip install -e ".[dev]" --no-build-isolation -vv
```

### Running unit tests

To run python tests run:

```bash
pytest
```

Some tests are marked as 'slow' and aren't run by default. You can use `pytest
-m slow` to run those, or `pytest -m ""` to run all tests, slow or not.

To run the C++ tests run:

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=1 -DCMAKE_PREFIX_PATH=$(python3 -c 'import torch;print(torch.utils.cmake_prefix_path)') ..
cmake --build . -- VERBOSE=1
ctest --rerun-failed --output-on-failure
```

### Code formatting and type checking

We use `pre-commit` to enforce code formatting and `mypy` for type checking.
Install both with

```bash
pip install pre-commit mypy
```

To run pre-commit hooks before each commit, run `pre-commit install`. You may
prefer to run these checks manually, in which case you can just use `pre-commit
run --all-files`.

For `mypy` we recommend the following command:

```bash
mypy --install-types --non-interactive --config-file mypy.ini
```

### Building the docs

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

## License

By contributing to TorchCodec, you agree that your contributions will be
licensed under the LICENSE file in the root directory of this source tree.

Contributors are also required to
[sign our Contributor License Agreement](https://code.facebook.com/cla).
