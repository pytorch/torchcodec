// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/torchcodec/_core/DeviceInterface.h"
#include <map>
#include <mutex>

namespace facebook::torchcodec {

namespace {
std::mutex g_interface_mutex;
std::map<std::string, CreateDeviceInterfaceFn> g_interface_map;

std::string getDeviceType(const std::string& device) {
  size_t pos = device.find(':');
  if (pos == std::string::npos) {
    return device;
  }
  return device.substr(0, pos);
}

} // namespace

bool registerDeviceInterface(
    const std::string deviceType,
    CreateDeviceInterfaceFn createInterface) {
  std::scoped_lock lock(g_interface_mutex);
  TORCH_CHECK(
      g_interface_map.find(deviceType) == g_interface_map.end(),
      "Device interface already registered for ",
      deviceType);
  g_interface_map.insert({deviceType, createInterface});
  return true;
}

std::unique_ptr<DeviceInterface> createDeviceInterface(
    const std::string device) {
  // TODO: remove once DeviceInterface for CPU is implemented
  if (device == "cpu") {
    return nullptr;
  }

  std::scoped_lock lock(g_interface_mutex);
  std::string deviceType = getDeviceType(device);
  TORCH_CHECK(
      g_interface_map.find(deviceType) != g_interface_map.end(),
      "Unsupported device: ",
      device);

  return std::unique_ptr<DeviceInterface>(g_interface_map[deviceType](device));
}

} // namespace facebook::torchcodec
