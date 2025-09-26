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
using DeviceInterfaceMap =
    std::map<DeviceInterfaceKey, CreateDeviceInterfaceFn>;
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
    const DeviceInterfaceKey& key,
    CreateDeviceInterfaceFn createInterface) {
  std::scoped_lock lock(g_interface_mutex);
  DeviceInterfaceMap& deviceMap = getDeviceMap();

  TORCH_CHECK(
      deviceMap.find(key) == deviceMap.end(),
      "Device interface already registered for device type ",
      key.deviceType,
      " variant '",
      key.variant,
      "'");
  deviceMap.insert({key, createInterface});

  return true;
}

bool registerDeviceInterface(
    torch::DeviceType deviceType,
    CreateDeviceInterfaceFn createInterface) {
  return registerDeviceInterface(
      DeviceInterfaceKey(deviceType), createInterface);
}

void validateDeviceInterface(
    const std::string device,
    const std::string variant) {
  std::scoped_lock lock(g_interface_mutex);
  std::string deviceType = getDeviceType(device);

  DeviceInterfaceMap& deviceMap = getDeviceMap();

  // Find device interface that matches device type and variant
  torch::DeviceType deviceTypeEnum = torch::Device(deviceType).type();

  auto deviceInterface = std::find_if(
      deviceMap.begin(),
      deviceMap.end(),
      [&](const std::pair<DeviceInterfaceKey, CreateDeviceInterfaceFn>& arg) {
        return arg.first.deviceType == deviceTypeEnum &&
            arg.first.variant == variant;
      });

  TORCH_CHECK(
      deviceInterface != deviceMap.end(),
      "Unsupported device: ",
      device,
      " (device type: ",
      deviceType,
      ", variant: ",
      variant,
      ")");
}

std::unique_ptr<DeviceInterface> createDeviceInterface(
    const torch::Device& device,
    const std::string_view variant) {
  DeviceInterfaceKey key(device.type(), variant);
  std::scoped_lock lock(g_interface_mutex);
  DeviceInterfaceMap& deviceMap = getDeviceMap();

  auto it = deviceMap.find(key);
  if (it != deviceMap.end()) {
    return std::unique_ptr<DeviceInterface>(it->second(device));
  }

  TORCH_CHECK(
      false,
      "No device interface found for device type: ",
      device.type(),
      " variant: '",
      variant,
      "'");
}

} // namespace facebook::torchcodec
