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
using DeviceInterfaceMap = std::map<torch::DeviceType, CreateDeviceInterfaceFn>;
std::mutex g_interface_mutex;
std::unique_ptr<DeviceInterfaceMap> g_interface_map;

std::string getDeviceType(const std::string& device) {
  size_t pos = device.find(':');
  if (pos == std::string::npos) {
    return device;
  }
  return device.substr(0, pos);
}

} // namespace

bool registerDeviceInterface(
    torch::DeviceType deviceType,
    CreateDeviceInterfaceFn createInterface) {
  std::scoped_lock lock(g_interface_mutex);
  if (!g_interface_map) {
    // We delay this initialization until runtime to avoid the Static
    // Initialization Order Fiasco:
    //
    //   https://en.cppreference.com/w/cpp/language/siof
    g_interface_map = std::make_unique<DeviceInterfaceMap>();
  }
  TORCH_CHECK(
      g_interface_map->find(deviceType) == g_interface_map->end(),
      "Device interface already registered for ",
      deviceType);
  g_interface_map->insert({deviceType, createInterface});
  return true;
}

torch::Device createTorchDevice(const std::string device) {
  // TODO: remove once DeviceInterface for CPU is implemented
  if (device == "cpu") {
    return torch::kCPU;
  }

  std::scoped_lock lock(g_interface_mutex);
  std::string deviceType = getDeviceType(device);
  auto deviceInterface = std::find_if(
      g_interface_map->begin(),
      g_interface_map->end(),
      [&](const std::pair<torch::DeviceType, CreateDeviceInterfaceFn>& arg) {
        return device.rfind(
                   torch::DeviceTypeName(arg.first, /*lcase*/ true), 0) == 0;
      });
  TORCH_CHECK(
      deviceInterface != g_interface_map->end(),
      "Unsupported device: ",
      device);

  return torch::Device(device);
}

std::unique_ptr<DeviceInterface> createDeviceInterface(
    const torch::Device& device) {
  auto deviceType = device.type();
  // TODO: remove once DeviceInterface for CPU is implemented
  if (deviceType == torch::kCPU) {
    return nullptr;
  }

  std::scoped_lock lock(g_interface_mutex);
  TORCH_CHECK(
      g_interface_map->find(deviceType) != g_interface_map->end(),
      "Unsupported device: ",
      device);

  return std::unique_ptr<DeviceInterface>(
      (*g_interface_map)[deviceType](device));
}

} // namespace facebook::torchcodec
