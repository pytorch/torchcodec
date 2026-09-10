#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This script installs PyTorch and other optional torch packages like
# torchvision from either the nightly or test channel based on the branch: test
# for release branches (and PRs against a release branch), nightly otherwise
#
# Example usage:
#   install_pytorch.sh cpu "torch torchvision"
#   install_pytorch.sh cpu "torch"
#   install_pytorch.sh cu126 "torch torchvision"

set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Error: Missing required arguments"
  echo "Usage: install_pytorch.sh COMPUTE_PLATFORM PACKAGES"
  echo "Example: install_pytorch.sh cpu \"torch torchvision\""
  exit 1
fi

COMPUTE_PLATFORM="$1"
PACKAGES="$2"

if [[ (${GITHUB_EVENT_NAME:-} = 'pull_request' && (${GITHUB_BASE_REF:-} = 'release'*)) || (${GITHUB_REF:-} = 'refs/heads/release'*) || (${GITHUB_REF:-} = refs/tags/v*) ]]; then
  CHANNEL=test
else
  CHANNEL=nightly
fi

echo "Installing PyTorch packages: $PACKAGES"
echo "Compute platform: $COMPUTE_PLATFORM"
echo "Channel: $CHANNEL"

python -m pip install --pre $PACKAGES --index-url https://download.pytorch.org/whl/${CHANNEL}/${COMPUTE_PLATFORM}
