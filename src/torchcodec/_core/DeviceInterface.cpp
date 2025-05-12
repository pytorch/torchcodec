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
static std::mutex g_interface_mutex;

DeviceInterfaceMap& getDeviceMap() {
  static DeviceInterfaceMap deviceMap;
  return deviceMap;
}

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
  DeviceInterfaceMap& deviceMap = getDeviceMap();

  TORCH_CHECK(
      deviceMap.find(deviceType) == deviceMap.end(),
      "Device interface already registered for ",
      deviceType);
  deviceMap.insert({deviceType, createInterface});

  return true;
}

torch::Device createTorchDevice(const std::string device) {
  std::scoped_lock lock(g_interface_mutex);
  std::string deviceType = getDeviceType(device);
  DeviceInterfaceMap& deviceMap = getDeviceMap();

  auto deviceInterface = std::find_if(
      deviceMap.begin(),
      deviceMap.end(),
      [&](const std::pair<torch::DeviceType, CreateDeviceInterfaceFn>& arg) {
        return device.rfind(
                   torch::DeviceTypeName(arg.first, /*lcase*/ true), 0) == 0;
      });
  TORCH_CHECK(
      deviceInterface != deviceMap.end(), "Unsupported device: ", device);

  return torch::Device(device);
}

std::unique_ptr<DeviceInterface> createDeviceInterface(
    const torch::Device& device) {
  auto deviceType = device.type();
  std::scoped_lock lock(g_interface_mutex);
  DeviceInterfaceMap& deviceMap = getDeviceMap();

  TORCH_CHECK(
      deviceMap.find(deviceType) != deviceMap.end(),
      "Unsupported device: ",
      device);

  return std::unique_ptr<DeviceInterface>(deviceMap[deviceType](device));
}

} // namespace facebook::torchcodec
