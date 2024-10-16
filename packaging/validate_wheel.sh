#!/usr/bin/env bash

set -eux

conda install ffmpeg -c conda-forge

# line below is inspired by:
#    https://github.com/pytorch/text/blob/1d4ce73c57417f1af8278af56631de3c25e3bbaf/.github/scripts/validate_binaries.sh#L5
# but when looking at the actual job that runs, we seem to install 0.0.0dev, a version
# we probably uploaded during release testing. How do we get the wheel generated during
# the build job?
pip install ${PYTORCH_PIP_PREFIX} torchcodec --extra-index-url ${PYTORCH_PIP_DOWNLOAD_URL}
pip install --pre torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
pip install numpy pytest pillow

python3 test/decoders/manual_smoke_test.py

pytest test -vvv
