// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "src/torchcodec/_core/DeviceInterface.h"

#include <memory>
#include <mutex>
#include <queue>
#include <unordered_map>

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
  bool canDecodePacketDirectly() const override {
    return true;
  }

  UniqueAVFrame decodePacketDirectly(ReferenceAVPacket& packet) override;

  // Legacy method name - kept for compatibility
  UniqueAVFrame decodePacket(ReferenceAVPacket& packet) {
    return decodePacketDirectly(packet);
  }

 public:
  // NVDEC callback functions (must be public for C callbacks)
  int handleVideoSequence(CUVIDEOFORMAT* pVideoFormat);
  int handlePictureDecode(CUVIDPICPARAMS* pPicParams);
  int handlePictureDisplay(CUVIDPARSERDISPINFO* pDispInfo);

 private:
  // NVDEC decoder context and parser
  CUvideoparser videoParser_ = nullptr;
  CUvideodecoder decoder_ = nullptr;
  CUcontext context_ = nullptr;

  // Video format info
  CUVIDEOFORMAT videoFormat_;
  CUVIDDECODECREATEINFO createInfo_;
  AVCodecID currentCodecId_ = AV_CODEC_ID_NONE;
  bool isInitialized_ = false;
  bool parserInitialized_ = false;

  // Frame queue for async decoding - stores frame pointer, pitch, and display info
  struct FrameData {
    CUdeviceptr framePtr;
    unsigned int pitch;
    CUVIDPARSERDISPINFO dispInfo;
  };
  std::queue<FrameData> frameQueue_;
  std::mutex frameQueueMutex_;

  // Custom context initialization for direct NVDEC usage
  void initializeNvdecDecoder(AVCodecID codecId);

  // Initialize video parser
  void initializeVideoParser(AVCodecID codecId, uint8_t* extradata, int extradata_size);

  // Convert CUDA frame pointer to AVFrame
  UniqueAVFrame convertCudaFrameToAVFrame(
      CUdeviceptr framePtr,
      unsigned int pitch,
      const CUVIDPARSERDISPINFO& dispInfo);
};

} // namespace facebook::torchcodec
