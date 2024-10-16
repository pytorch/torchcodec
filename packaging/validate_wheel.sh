#!/usr/bin/env bash

set -eux

conda install ffmpeg -c conda-forge
pip install ${PYTORCH_PIP_PREFIX} torchcodec --extra-index-url ${PYTORCH_PIP_DOWNLOAD_URL}
pip install --pre torchvision --index-url https://download.pytorch.org/whl/nightly/cpu

python3 test/decoders/manual_smoke_test.py
pytest test -vvv
