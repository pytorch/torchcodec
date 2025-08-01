// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "src/torchcodec/_core/DeviceInterface.h"

#include <queue>
#include <mutex>
#include <unordered_map>
#include <memory>

// Include NVIDIA Video Codec SDK headers
#include <cuviddec.h>
#include <nvcuvid.h>

namespace facebook::torchcodec {

// Cache key for decoder and context reuse
struct NvdecCacheKey {
  int deviceId;
  cudaVideoCodec codec;
  int width;
  int height;
  cudaVideoChromaFormat chromaFormat;
  int bitDepthLumaMinus8;
  int bitDepthChromaMinus8;

  bool operator==(const NvdecCacheKey& other) const {
    return deviceId == other.deviceId &&
           codec == other.codec &&
           width == other.width &&
           height == other.height &&
           chromaFormat == other.chromaFormat &&
           bitDepthLumaMinus8 == other.bitDepthLumaMinus8 &&
           bitDepthChromaMinus8 == other.bitDepthChromaMinus8;
  }
};

// Hash function for NvdecCacheKey
struct NvdecCacheKeyHash {
  std::size_t operator()(const NvdecCacheKey& key) const {
    std::size_t h1 = std::hash<int>{}(key.deviceId);
    std::size_t h2 = std::hash<int>{}(static_cast<int>(key.codec));
    std::size_t h3 = std::hash<int>{}(key.width);
    std::size_t h4 = std::hash<int>{}(key.height);
    std::size_t h5 = std::hash<int>{}(static_cast<int>(key.chromaFormat));
    std::size_t h6 = std::hash<int>{}(key.bitDepthLumaMinus8);
    std::size_t h7 = std::hash<int>{}(key.bitDepthChromaMinus8);
    return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3) ^ (h5 << 4) ^ (h6 << 5) ^ (h7 << 6);
  }
};

// Cached decoder and context objects
struct CachedNvdecObjects {
  CUvideodecoder decoder;
  CUcontext context;
  int refCount;
  
  CachedNvdecObjects(CUvideodecoder dec, CUcontext ctx) 
    : decoder(dec), context(ctx), refCount(1) {}
};

// Global cache for NVDEC decoders and contexts
class NvdecCache {
public:
  static NvdecCache& getInstance() {
    static NvdecCache instance;
    return instance;
  }

  std::shared_ptr<CachedNvdecObjects> getOrCreate(
      const NvdecCacheKey& key,
      const CUVIDEOFORMAT& videoFormat);
  
  void release(const NvdecCacheKey& key);
  void cleanup(); // For testing/cleanup

private:
  std::unordered_map<NvdecCacheKey, std::shared_ptr<CachedNvdecObjects>, NvdecCacheKeyHash> cache_;
  std::mutex cacheMutex_;
  
  NvdecCache() = default;
  ~NvdecCache() { cleanup(); }
  
  // Non-copyable
  NvdecCache(const NvdecCache&) = delete;
  NvdecCache& operator=(const NvdecCache&) = delete;
};

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
  std::shared_ptr<CachedNvdecObjects> cachedObjects_ = nullptr;
  NvdecCacheKey cacheKey_;

  // Video format info
  CUVIDEOFORMAT videoFormat_;
  AVCodecID currentCodecId_ = AV_CODEC_ID_NONE;
  bool isInitialized_ = false;
  bool parserInitialized_ = false;

  // Frame queue for async decoding
  std::queue<std::pair<CUdeviceptr, CUVIDPARSERDISPINFO>> frameQueue_;
  std::mutex frameQueueMutex_;

  // Custom context initialization for direct NVDEC usage
  void initializeNvdecDecoder(AVCodecID codecId);

  // Initialize video parser
  void initializeVideoParser(AVCodecID codecId);

  // Convert NVDEC output to AVFrame for compatibility with existing pipeline
  UniqueAVFrame convertNvdecOutputToAVFrame(
      uint8_t* decodedFrame,
      int width,
      int height,
      int64_t pts,
      int64_t duration);

  // Convert CUDA frame pointer to AVFrame
  UniqueAVFrame convertCudaFrameToAVFrame(
      CUdeviceptr framePtr,
      const CUVIDPARSERDISPINFO& dispInfo);
};

} // namespace facebook::torchcodec
