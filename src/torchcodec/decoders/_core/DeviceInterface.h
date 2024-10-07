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

// Note that all these device functions should only be called if the device is
// not a CPU device. CPU device functions are already implemented in the
// VideoDecoder implementation.

// Initialize the hardware device that is specified in `device`. Some builds
// support CUDA and others only support CPU.
void initializeDeviceContext(const torch::Device& device);

} // namespace facebook::torchcodec
