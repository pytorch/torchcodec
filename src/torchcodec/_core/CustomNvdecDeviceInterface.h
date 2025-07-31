// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "src/torchcodec/_core/DeviceInterface.h"

// Include NVIDIA Video Codec SDK headers
#include <cuviddec.h>
#include <nvcuvid.h>

namespace facebook::torchcodec {

// Custom NVDEC device interface that provides direct control over NVDEC
// while keeping FFmpeg for demuxing
class CustomNvdecDeviceInterface : public DeviceInterface {
 public:
  CustomNvdecDeviceInterface(const torch::Device& device);

  virtual ~CustomNvdecDeviceInterface();

  std::optional<const AVCodec*> findCodec(const AVCodecID& codecId) override;

  void initializeContext(AVCodecContext* codecContext) override;

  void convertAVFrameToFrameOutput(
      const VideoStreamOptions& videoStreamOptions,
      const AVRational& timeBase,
      UniqueAVFrame& avFrame,
      FrameOutput& frameOutput,
      std::optional<torch::Tensor> preAllocatedOutputTensor =
          std::nullopt) override;

  // Extension point overrides for direct packet decoding
  bool canDecodePacketDirectly() const override { return true; }
  
  UniqueAVFrame decodePacketDirectly(ReferenceAVPacket& packet) override;

  // Legacy method name - kept for compatibility
  UniqueAVFrame decodePacket(ReferenceAVPacket& packet) {
    return decodePacketDirectly(packet);
  }

 private:
  // NVDEC decoder context and parser
  CUvideoparser videoParser_ = nullptr;
  CUvideodecoder videoDecoder_ = nullptr;
  CUcontext cudaContext_ = nullptr;
  
  // Video format info
  CUVIDEOFORMAT videoFormat_;
  bool isInitialized_ = false;
  
  // Custom context initialization for direct NVDEC usage
  void initializeNvdecDecoder(AVCodecID codecId);
  
  // Convert NVDEC output to AVFrame for compatibility with existing pipeline
  UniqueAVFrame convertNvdecOutputToAVFrame(
      uint8_t* decodedFrame, 
      int width, 
      int height, 
      int64_t pts, 
      int64_t duration);
};

} // namespace facebook::torchcodec