// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/types.h>
#include <memory>
#include <stdexcept>
#include <string>

namespace facebook::torchcodec {

// Initialize the hardware device that is specified in `device`. Some builds
// support CUDA and others only support CPU.
void maybeInitializeDeviceContext(const torch::Device& device);

} // namespace facebook::torchcodec
