// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// BETA CUDA device interface that provides direct control over NVDEC
// while keeping FFmpeg for demuxing. A lot of the logic, particularly the use
// of a cache for the decoders, is inspired by DALI's implementation which is
// APACHE 2.0:
// https://github.com/NVIDIA/DALI/blob/c7539676a24a8e9e99a6e8665e277363c5445259/dali/operators/video/frames_decoder_gpu.cc#L1
//
// NVDEC / NVCUVID docs:
// https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvdec-video-decoder-api-prog-guide/index.html#using-nvidia-video-decoder-nvdecode-api

#pragma once

#include "src/torchcodec/_core/Cache.h"
#include "src/torchcodec/_core/DeviceInterface.h"
#include "src/torchcodec/_core/FFMPEGCommon.h"
#include "src/torchcodec/_core/NVDECCache.h"

#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <vector>

#include "src/torchcodec/_core/nvcuvid_include/cuviddec.h"
#include "src/torchcodec/_core/nvcuvid_include/nvcuvid.h"

namespace facebook::torchcodec {

class BetaCudaDeviceInterface : public DeviceInterface {
 public:
  explicit BetaCudaDeviceInterface(const torch::Device& device);
  virtual ~BetaCudaDeviceInterface();

  void initializeInterface(AVStream* stream) override;

  void convertAVFrameToFrameOutput(
      const VideoStreamOptions& videoStreamOptions,
      const AVRational& timeBase,
      UniqueAVFrame& avFrame,
      FrameOutput& frameOutput,
      std::optional<torch::Tensor> preAllocatedOutputTensor =
          std::nullopt) override;

  bool canDecodePacketDirectly() const override {
    return true;
  }

  int sendPacket(ReferenceAVPacket& packet) override;
  int receiveFrame(UniqueAVFrame& avFrame, int64_t desiredPts) override;
  void flush() override;
  ReferenceAVPacket* applyBSF(
      ReferenceAVPacket& packet,
      AutoAVPacket& filteredAutoPacket,
      ReferenceAVPacket& filteredPacket) override;

  // NVDEC callback functions (must be public for C callbacks)
  unsigned char streamPropertyChange(CUVIDEOFORMAT* videoFormat);
  int frameReadyForDecoding(CUVIDPICPARAMS* pPicParams);

 private:
  UniqueAVFrame convertCudaFrameToAVFrame(
      CUdeviceptr framePtr,
      unsigned int pitch,
      const CUVIDPARSERDISPINFO& dispInfo);

  CUvideoparser videoParser_ = nullptr;
  UniqueCUvideodecoder decoder_;
  CUVIDEOFORMAT videoFormat_ = {};

  struct FrameBufferSlot {
    CUVIDPARSERDISPINFO dispInfo;
    int64_t guessedPts;
    bool occupied = false;

    FrameBufferSlot() : guessedPts(-1), occupied(false) {
      memset(&dispInfo, 0, sizeof(dispInfo));
    }
  };

  std::vector<FrameBufferSlot> frameBuffer_;
  FrameBufferSlot* findEmptySlot();
  FrameBufferSlot* findFrameWithExactPts(int64_t desiredPts);

  std::queue<int64_t> packetsPtsQueue;

  bool eofSent_ = false;

  // Flush flag to prevent decode operations during flush (like DALI's
  // isFlushing_)
  bool isFlushing_ = false;

  AVRational timeBase_ = {0, 0};

  UniqueAVBSFContext bitstreamFilter_;

  // Default CUDA interface for color conversion.
  // TODONVDEC P2: we shouldn't need to keep a separate instance of the default.
  // See other TODO there about how interfaces should be completely independent.
  std::unique_ptr<DeviceInterface> defaultCudaInterface_;
};

} // namespace facebook::torchcodec
