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
- ninja
- pkg-config
- scikit-build-core (our build backend)
- FFmpeg
- (Optional) image codec libraries - see image decoder section below
- PyTorch nightly

Start by installing the **nightly** build of PyTorch following the
[official instructions](https://pytorch.org/get-started/locally/).

Then, the easiest way to install the rest of the dependencies is to run:

```bash
conda install cmake ninja pkg-config scikit-build-core "ffmpeg" -c conda-forge
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

### Image decoder build options

By default, TorchCodec builds all of its image decoders (currently jpeg, png,
avif, webp, and gif) and **requires** them: if a codec's library can't be found,
the build fails.

The libraries that need to be externally sourced are libjpeg, libpng, and
libwebp (TorchCodec will provide libavif and giflib). You can typically install
them via `conda install libjpeg-turbo libpng libwebp -c conda-forge`.

You can opt-out from building all the image decoders by setting
`TORCHCODEC_BUILD_IMAGE=0`, or from building individual image decoders by
setting one of `TORCHCODEC_BUILD_{JPEG,PNG,WEBP,AVIF,GIF}=0`. To build a single
image decoder, e.g. jpeg, set `TORCHCODEC_BUILD_IMAGE=0
TORCHCODEC_BUILD_JPEG=1`.

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

### Serving docs locally (if building from a GPU env)

If you're developing locally, you can just open the generated `index.html`file
in your browser.

If instead you're using a remote machine, you can use a combination of a simple
python HTTP server and port forwarding to serve the docs locally. This allows
you to iterate on the documentation much more quickly than relying on
PR previews.

To do so, after following the above doc build steps, run the following from
the `docs/build/html` folder:

```
python -m http.server 8000 # or any free port
```

This will open up a simple HTTP server serving the files in the build directory.
If this is done on a remote machine, you can set up port forwarding from your
local machine to access the server, for example:

```
ssh -L 9000:localhost:8000 $REMOTE_DEV_HOST
```

Now, you can navigate to `localhost:9000` on your local machine to view the
rendered documentation.

## License

By contributing to TorchCodec, you agree that your contributions will be
licensed under the LICENSE file in the root directory of this source tree.

Contributors are also required to
[sign our Contributor License Agreement](https://code.facebook.com/cla).
