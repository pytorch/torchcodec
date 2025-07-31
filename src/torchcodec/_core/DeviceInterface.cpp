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
using DeviceInterfaceMap = std::map<DeviceInterfaceKey, CreateDeviceInterfaceFn>;
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

// New registration function with variant support
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

// Backward-compatible registration function
bool registerDeviceInterface(
    torch::DeviceType deviceType,
    CreateDeviceInterfaceFn createInterface) {
  return registerDeviceInterface(DeviceInterfaceKey(deviceType), createInterface);
}

torch::Device createTorchDevice(const std::string device) {
  std::scoped_lock lock(g_interface_mutex);
  std::string deviceType = getDeviceType(device);
  DeviceInterfaceMap& deviceMap = getDeviceMap();

  auto deviceInterface = std::find_if(
      deviceMap.begin(),
      deviceMap.end(),
      [&](const std::pair<DeviceInterfaceKey, CreateDeviceInterfaceFn>& arg) {
        return device.rfind(
                   torch::DeviceTypeName(arg.first.deviceType, /*lcase*/ true), 0) == 0;
      });
  TORCH_CHECK(
      deviceInterface != deviceMap.end(), "Unsupported device: ", device);

  return torch::Device(device);
}

// Creation function with variant support (default = "default" for backward compatibility)
std::unique_ptr<DeviceInterface> createDeviceInterface(
    const torch::Device& device,
    const std::string& variant) {
  DeviceInterfaceKey key(device.type(), variant);
  std::scoped_lock lock(g_interface_mutex);
  DeviceInterfaceMap& deviceMap = getDeviceMap();

  auto it = deviceMap.find(key);
  if (it != deviceMap.end()) {
    return std::unique_ptr<DeviceInterface>(it->second(device));
  }
  
  // Fallback to default variant if specific variant not found
  if (variant != "default") {
    key.variant = "default";
    it = deviceMap.find(key);
    if (it != deviceMap.end()) {
      return std::unique_ptr<DeviceInterface>(it->second(device));
    }
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
