// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/torchcodec/decoders/_core/DeviceInterface.h"
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
    const std::string device_type,
    CreateDeviceInterfaceFn create_interface) {
  std::scoped_lock lock(g_interface_mutex);
  TORCH_CHECK(
      g_interface_map.find(device_type) == g_interface_map.end(),
      "Device interface already registered for ",
      device_type);
  g_interface_map.insert({device_type, create_interface});
  return true;
}

std::shared_ptr<DeviceInterface> createDeviceInterface(
    const std::string device) {
  // TODO: remove once DeviceInterface for CPU is implemented
  if (device == "cpu") {
    return nullptr;
    // return std::shared_ptr<DeviceInterface>();
  }

  std::scoped_lock lock(g_interface_mutex);
  std::string device_type = getDeviceType(device);
  TORCH_CHECK(
      g_interface_map.find(device_type) != g_interface_map.end(),
      "Unsupported device: ",
      device);

  return std::shared_ptr<DeviceInterface>(g_interface_map[device_type](device));
  // return std::shared_ptr<DeviceInterface>(g_interface_map[device_type]);
}

} // namespace facebook::torchcodec
