#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Installs the rocJPEG SDK
#
# rocJPEG is not preinstalled by default in the CI runners. Both the build
# and the test jobs need it (we don't ship it in the wheel, unlike nvjpeg), so
# we install it from the ROCm dnf repo.

set -euo pipefail

# rocJPEG decodes via VA-API, so at runtime it needs the AMD VA-API driver
# (mesa-amdgpu-va-drivers + libva-amdgpu), not just librocjpeg -- without it,
# vaInitialize() fails and decoding errors out. A proper `dnf install` pulls that
# whole stack, but only where AMD's amdgpu-graphics repo is configured (the ROCm
# test runners). On the plain build image that repo is absent; there we only need
# to *compile* against rocjpeg.h/librocjpeg.so, so fall back to installing just
# those with --nodeps (base libva provides the libva.so.2 soname we link).
install_rocjpeg_build_only() {
    dnf install -y --refresh libva
    dnf install -y "dnf-command(download)" >/dev/null 2>&1 || dnf install -y dnf-plugins-core
    rpm_dir="$(mktemp -d)"
    dnf download --destdir "${rpm_dir}" rocjpeg rocjpeg-devel
    rpm -Uvh --nodeps "${rpm_dir}"/rocjpeg*.rpm
}

dnf install -y --refresh rocjpeg-devel libva-amdgpu mesa-amdgpu-va-drivers \
    || install_rocjpeg_build_only
