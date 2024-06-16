TorchCodec
==========

Welcome to TorchCodec!

TODO: Write a decent readme. For now I'm just writing basic info for OSS
development.


Installing from source
----------------------

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

Building the docs
-----------------

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

Run `make clean` from time to time if you encounter issues.
