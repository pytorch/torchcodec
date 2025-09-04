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

#include "src/torchcodec/_core/nvcuvid_include/cuviddec.h"
#include "src/torchcodec/_core/nvcuvid_include/nvcuvid.h"

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

  // New send/receive API (FFmpeg-style)
  // Send packet for decoding (non-blocking)
  // Returns 0 on success, AVERROR(EAGAIN) if decoder queue full, or other AVERROR on failure
  int sendPacket(ReferenceAVPacket& packet);

  // Receive decoded frame (non-blocking) 
  // Returns 0 on success, AVERROR(EAGAIN) if no frame ready, AVERROR_EOF if end of stream,
  // or other AVERROR on failure
  int receiveFrame(UniqueAVFrame& frame);

  // Flush remaining frames from decoder
  void flush();

 public:
  // NVDEC callback functions (must be public for C callbacks)
  int handleVideoSequence(CUVIDEOFORMAT* pVideoFormat);
  int handlePictureDecode(CUVIDPICPARAMS* pPicParams);
  int handlePictureDisplay(CUVIDPARSERDISPINFO* pDispInfo);

 private:
  // NVDEC decoder context and parser
  CUvideoparser videoParser_ = nullptr;
  CUvideodecoder decoder_ = nullptr;

  // Video format info
  CUVIDEOFORMAT videoFormat_;
  bool parserCreated_ = false;

  // Frame buffer for B-frame reordering (like DALI)
  struct BufferedFrame {
    CUVIDPARSERDISPINFO dispInfo;
    int64_t pts;
    bool available = false;
    
    BufferedFrame() : pts(-1), available(false) {
      memset(&dispInfo, 0, sizeof(dispInfo));
    }
  };
  
  static constexpr int MAX_DECODE_SURFACES = 32; // NVDEC max
  std::vector<BufferedFrame> frameBuffer_;
  std::mutex frameBufferMutex_;

  // PTS queue for proper packet-to-frame mapping (like DALI)
  std::queue<int64_t> pipedPts_;

  // EOF tracking
  bool eofSent_ = false;

  // Helper methods for frame reordering
  BufferedFrame* findEmptySlot();
  BufferedFrame* findFrameWithEarliestPts();


  // Initialize video parser
  void createVideoParser();

  // Convert CUDA frame pointer to AVFrame
  UniqueAVFrame convertCudaFrameToAVFrame(
      CUdeviceptr framePtr,
      unsigned int pitch,
      const CUVIDPARSERDISPINFO& dispInfo);
};

} // namespace facebook::torchcodec
