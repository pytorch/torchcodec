// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/types.h>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include "FFMPEGCommon.h"
#include "src/torchcodec/_core/Frame.h"
#include "src/torchcodec/_core/StreamOptions.h"

namespace facebook::torchcodec {

// Key for device interface registration with device type + variant support
struct DeviceInterfaceKey {
  torch::DeviceType deviceType;
  std::string variant = "default"; // e.g., "default", "custom_nvdec", etc.

  bool operator<(const DeviceInterfaceKey& other) const {
    if (deviceType != other.deviceType) {
      return deviceType < other.deviceType;
    }
    return variant < other.variant;
  }

  // Convenience constructors
  DeviceInterfaceKey(torch::DeviceType type) : deviceType(type) {}

  DeviceInterfaceKey(torch::DeviceType type, const std::string& var)
      : deviceType(type), variant(var) {}
};

class DeviceInterface {
 public:
  DeviceInterface(const torch::Device& device) : device_(device) {}

  virtual ~DeviceInterface(){};

  torch::Device& device() {
    return device_;
  };

  virtual std::optional<const AVCodec*> findCodec(const AVCodecID& codecId) = 0;

  // Initialize the hardware device that is specified in `device`. Some builds
  // support CUDA and others only support CPU.
  virtual void initializeContext(AVCodecContext* codecContext) = 0;

  virtual void convertAVFrameToFrameOutput(
      const VideoStreamOptions& videoStreamOptions,
      const AVRational& timeBase,
      UniqueAVFrame& avFrame,
      FrameOutput& frameOutput,
      std::optional<torch::Tensor> preAllocatedOutputTensor = std::nullopt) = 0;

  // Extension points for custom decoding paths
  // Override to return true if this device interface can decode packets
  // directly
  virtual bool canDecodePacketDirectly() const {
    return false;
  }

  // Override to decode AVPacket directly (bypassing FFmpeg codec)
  // Only called if canDecodePacketDirectly() returns true
  virtual UniqueAVFrame decodePacketDirectly(ReferenceAVPacket& /* packet */) {
    TORCH_CHECK(
        false,
        "Direct packet decoding not implemented for this device interface");
    return UniqueAVFrame(nullptr);
  }

  // New send/receive API for custom decoders (FFmpeg-style)
  // Send packet for decoding (non-blocking)
  // Returns 0 on success, AVERROR(EAGAIN) if decoder queue full, or other AVERROR on failure
  virtual int sendPacket(ReferenceAVPacket& /* packet */) {
    TORCH_CHECK(
        false,
        "Send/receive packet decoding not implemented for this device interface");
    return AVERROR(ENOSYS);
  }

  // Receive decoded frame (non-blocking) 
  // Returns 0 on success, AVERROR(EAGAIN) if no frame ready, AVERROR_EOF if end of stream,
  // or other AVERROR on failure
  virtual int receiveFrame(UniqueAVFrame& /* frame */) {
    TORCH_CHECK(
        false,
        "Send/receive packet decoding not implemented for this device interface");
    return AVERROR(ENOSYS);
  }

  // Flush remaining frames from decoder
  virtual void flush() {
    // Default implementation is no-op for standard decoders
    // Custom decoders can override this method
  }

 protected:
  torch::Device device_;
};

using CreateDeviceInterfaceFn =
    std::function<DeviceInterface*(const torch::Device& device)>;

// New registration function with variant support
bool registerDeviceInterface(
    const DeviceInterfaceKey& key,
    const CreateDeviceInterfaceFn createInterface);

// Backward-compatible registration function
bool registerDeviceInterface(
    torch::DeviceType deviceType,
    const CreateDeviceInterfaceFn createInterface);

torch::Device createTorchDevice(const std::string device);

// Creation function with variant support (default = "default" for backward
// compatibility)
std::unique_ptr<DeviceInterface> createDeviceInterface(
    const torch::Device& device,
    const std::string& variant = "default");

} // namespace facebook::torchcodec
