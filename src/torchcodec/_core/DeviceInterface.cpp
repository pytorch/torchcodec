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
  return registerDeviceInterface(
      DeviceInterfaceKey(deviceType), createInterface);
}

torch::Device createTorchDevice(const std::string device) {
  std::scoped_lock lock(g_interface_mutex);
  
  // Parse device string: "device_type:index:variant" or "device_type:index" or "device_type"
  std::string deviceType;
  std::string variant = "default";
  std::string torchDeviceString = device; // What we'll pass to torch::Device constructor
  
  size_t firstColon = device.find(':');
  if (firstColon == std::string::npos) {
    // Just device type (e.g., "cpu")
    deviceType = device;
  } else {
    deviceType = device.substr(0, firstColon);
    
    // Check for second colon (variant)
    size_t secondColon = device.find(':', firstColon + 1);
    if (secondColon != std::string::npos) {
      // Format: "device_type:index:variant"
      variant = device.substr(secondColon + 1);
      torchDeviceString = device.substr(0, secondColon); // Remove variant part
    }
    // else: Format: "device_type:index" (no variant)
  }
  
  DeviceInterfaceMap& deviceMap = getDeviceMap();

  // Find device interface that matches device type and variant
  torch::DeviceType deviceTypeEnum = torch::Device(deviceType).type();
  
  auto deviceInterface = std::find_if(
      deviceMap.begin(),
      deviceMap.end(),
      [&](const std::pair<DeviceInterfaceKey, CreateDeviceInterfaceFn>& arg) {
        return arg.first.deviceType == deviceTypeEnum && arg.first.variant == variant;
      });
      
  // If variant-specific interface not found, try default variant
  if (deviceInterface == deviceMap.end() && variant != "default") {
    deviceInterface = std::find_if(
        deviceMap.begin(),
        deviceMap.end(),
        [&](const std::pair<DeviceInterfaceKey, CreateDeviceInterfaceFn>& arg) {
          return arg.first.deviceType == deviceTypeEnum && arg.first.variant == "default";
        });
  }
  
  TORCH_CHECK(
      deviceInterface != deviceMap.end(), 
      "Unsupported device: ", device, 
      " (device type: ", deviceType, ", variant: ", variant, ")");

  // Return torch::Device with just device type and index (no variant)
  return torch::Device(torchDeviceString);
}

// Creation function with variant support (default = "default" for backward
// compatibility)
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
