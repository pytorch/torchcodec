// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/torchcodec/_core/CpuDeviceInterface.h"

namespace facebook::torchcodec {
namespace {

bool g_cpu = registerDeviceInterface(
    torch::kCPU,
    [](const torch::Device& device, const AVRational& timeBase) {
      return new CpuDeviceInterface(device, timeBase);
    });

} // namespace

CpuDeviceInterface::CpuDeviceInterface(
    const torch::Device& device,
    const AVRational& timeBase)
    : DeviceInterface(device, timeBase) {
  if (device_.type() != torch::kCPU) {
    throw std::runtime_error("Unsupported device: " + device_.str());
  }
}

} // namespace facebook::torchcodec
