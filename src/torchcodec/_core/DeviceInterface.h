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
  std::string_view variant = "default"; // e.g., "default", "beta", etc.

  bool operator<(const DeviceInterfaceKey& other) const {
    if (deviceType != other.deviceType) {
      return deviceType < other.deviceType;
    }
    return variant < other.variant;
  }

  explicit DeviceInterfaceKey(torch::DeviceType type) : deviceType(type) {}

  DeviceInterfaceKey(torch::DeviceType type, const std::string_view& var)
      : deviceType(type), variant(var) {}
};

class DeviceInterface {
 public:
  DeviceInterface(const torch::Device& device) : device_(device) {}

  virtual ~DeviceInterface(){};

  torch::Device& device() {
    return device_;
  };

  virtual std::optional<const AVCodec*> findCodec(
      [[maybe_unused]] const AVCodecID& codecId) {
    return std::nullopt;
  };

  // Initialize the hardware device that is specified in `device`. Some builds
  // support CUDA and others only support CPU.
  virtual void initializeContext(
      [[maybe_unused]] AVCodecContext* codecContext) {}

  virtual void initializeInterface([[maybe_unused]] AVStream* stream) {}

  virtual void convertAVFrameToFrameOutput(
      const VideoStreamOptions& videoStreamOptions,
      const AVRational& timeBase,
      UniqueAVFrame& avFrame,
      FrameOutput& frameOutput,
      std::optional<torch::Tensor> preAllocatedOutputTensor = std::nullopt) = 0;

  // ------------------------------------------
  // Extension points for custom decoding paths
  // ------------------------------------------

  // Override to return true if this device interface can decode packets
  // directly
  virtual bool canDecodePacketDirectly() const {
    return false;
  }

  // Moral equivalent of avcodec_send_packet()
  // Returns AVSUCCESS on success, AVERROR(EAGAIN) if decoder queue full, or
  // other AVERROR on failure
  virtual int sendPacket([[maybe_unused]] ReferenceAVPacket& avPacket) {
    TORCH_CHECK(
        false,
        "Send/receive packet decoding not implemented for this device interface");
    return AVERROR(ENOSYS);
  }

  // Moral equivalent of avcodec_receive_frame()
  // Returns AVSUCCESS on success, AVERROR(EAGAIN) if no frame ready,
  // AVERROR_EOF if end of stream, or other AVERROR on failure
  virtual int receiveFrame(
      [[maybe_unused]] UniqueAVFrame& avFrame,
      [[maybe_unused]] int64_t desiredPts) {
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

  // Apply bitstream filter if needed, returns pointer to packet to use
  // Default implementation returns the original packet (no filtering)
  virtual ReferenceAVPacket* applyBSF(
      ReferenceAVPacket& packet,
      [[maybe_unused]] AutoAVPacket& filteredAutoPacket,
      [[maybe_unused]] ReferenceAVPacket& filteredPacket) {
    return &packet; // No filtering by default
  }

 protected:
  torch::Device device_;
};

using CreateDeviceInterfaceFn =
    std::function<DeviceInterface*(const torch::Device& device)>;

bool registerDeviceInterface(
    const DeviceInterfaceKey& key,
    const CreateDeviceInterfaceFn createInterface);

// Backward-compatible registration function when variant is "default"
// TODONVDEC P2 We only need this if someone in the wild has already started
// registering their own interfaces. Ask Dmitry.
bool registerDeviceInterface(
    torch::DeviceType deviceType,
    const CreateDeviceInterfaceFn createInterface);

void validateDeviceInterface(
    const std::string device,
    const std::string variant);

std::unique_ptr<DeviceInterface> createDeviceInterface(
    const torch::Device& device,
    const std::string_view variant = "default");

} // namespace facebook::torchcodec
