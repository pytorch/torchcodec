#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Installs the build-time dependencies needed to *compile* torchcodec: the
# scikit-build-core build backend and the Ninja generator it drives.
# Because we build with --no-build-isolation (torch must come from a custom index,
# so it can't live in pyproject's build-system.requires), pip does not install
# build-system.requires for us -- we must do it here.
#
# Also installs the image decoder libs (libjpeg-turbo, libpng, libwebp, and
# libheif) by default. Builds that don't need image decoding (e.g. the mypy CI
# job) can skip them by setting TORCHCODEC_BUILD_IMAGE=0.
#
# This script intentionally does NOT install:
# - torch: install it separately (a nightly matching your CPU/CUDA variant).
# - FFmpeg: a runtime/build dependency handled by the caller.
# - libavif: unlike the other image libs, it is not installed from conda. The
#   build always fetches a slim decode-only libavif from S3
# - pkg-config: only needed to *locate an installed FFmpeg* at build time, i.e.
#   builds that don't set BUILD_AGAINST_ALL_FFMPEG_FROM_S3. Callers that build
#   against an installed FFmpeg install pkg-config alongside it.

set -ex

python -m pip install "scikit-build-core>=0.10" ninja

if [[ "${TORCHCODEC_BUILD_IMAGE:-ON}" != "0" ]]; then
    conda install -y libjpeg-turbo -c pytorch
    conda install -y libpng "libwebp>=1.3" libheif -c conda-forge
fi
