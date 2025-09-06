// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

// Debug flag - set to 1 to enable debug output, 0 to disable
#define CUSTOM_NVDEC_DEBUG 0

#include "src/torchcodec/_core/DeviceInterface.h"
#include "src/torchcodec/_core/Cache.h"
#include "src/torchcodec/_core/FFMPEGCommon.h"

#include <memory>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <vector>
#include <map>

#include "src/torchcodec/_core/nvcuvid_include/cuviddec.h"
#include "src/torchcodec/_core/nvcuvid_include/nvcuvid.h"

namespace facebook::torchcodec {

// Custom deleter for CUvideodecoder
struct CUvideoDecoderDeleter {
  void operator()(CUvideodecoder decoder) const {
    if (decoder) {
      cuvidDestroyDecoder(decoder);
    }
  }
};

using UniqueCUvideodecoder = std::unique_ptr<void, CUvideoDecoderDeleter>;

// Simple decoder key for parameter-based matching
struct NVDECDecoderKey {
  cudaVideoCodec codec_type;
  unsigned width;
  unsigned height;
  cudaVideoChromaFormat chroma_format;
  unsigned int bit_depth_luma_minus8;
  unsigned num_decode_surfaces;
  
  bool operator<(const NVDECDecoderKey& other) const {
    return std::tie(codec_type, width, height, chroma_format, bit_depth_luma_minus8, num_decode_surfaces) <
           std::tie(other.codec_type, other.width, other.height, other.chroma_format, other.bit_depth_luma_minus8, other.num_decode_surfaces);
  }
};

// Simple cache for NVDEC decoders using existing Cache.h patterns
class NVDECCache {
 public:
  // Get cache instance for specific device
  static NVDECCache& GetCache(int device_id = -1);

  // Get decoder from cache - returns nullptr if none available
  UniqueCUvideodecoder getDecoder(const NVDECDecoderKey& key);

  // Return decoder to cache - returns true if added to cache
  bool returnDecoder(const NVDECDecoderKey& key, UniqueCUvideodecoder decoder);

  // Create new decoder with given parameters
  static UniqueCUvideodecoder createDecoder(CUVIDEOFORMAT* video_format);

  // Helper to create key from video format
  static NVDECDecoderKey createKey(CUVIDEOFORMAT* video_format);

 private:
  NVDECCache() = default;
  ~NVDECCache() = default;

  std::map<NVDECDecoderKey, UniqueCUvideodecoder> cache_;
  std::mutex cache_lock_;
  
  static constexpr int MAX_CACHE_SIZE = 20; // Much smaller, simpler cache
};


// Custom NVDEC device interface that provides direct control over NVDEC
// while keeping FFmpeg for demuxing
class CustomNvdecDeviceInterface : public DeviceInterface {
 public:
  CustomNvdecDeviceInterface(const torch::Device& device);

  virtual ~CustomNvdecDeviceInterface();

  std::optional<const AVCodec*> findCodec(const AVCodecID& codecId) override;

  void initializeContext(AVCodecContext* codecContext) override;

  // Set the timeBase for duration calculations
  void setTimeBase(const AVRational& timeBase);

  // Set the frame rate for duration calculations
  void setFrameRate(const AVRational& frameRate);

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

 private:
  // NVDEC decoder context and parser
  CUvideoparser videoParser_ = nullptr;
  UniqueCUvideodecoder decoder_;
  NVDECDecoderKey decoderKey_; // Store key for return to cache

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

  // PTS priority queue for proper packet-to-frame mapping (like DALI)
  // Using min-heap to efficiently get smallest PTS
  std::priority_queue<int64_t, std::vector<int64_t>, std::greater<int64_t>> pipedPts_;

  // Decode surface tracking (like DALI's frame_in_use_)
  std::vector<uint8_t> surfaceInUse_;

  // EOF tracking
  bool eofSent_ = false;
  
  // Current PTS being processed (like DALI's current_pts_)
  int64_t currentPts_ = AV_NOPTS_VALUE;
  
  // Flush flag to prevent decode operations during flush (like DALI's flush_)
  bool flush_ = false;
  
  // Store timeBase for duration calculations
  AVRational timeBase_ = {0, 0};
  
  // Store frame rate for duration calculations (fallback when NVDEC frame rate is unavailable)
  AVRational fallbackFrameRate_ = {0, 0};

  // Helper methods for frame reordering
  BufferedFrame* findEmptySlot();
  BufferedFrame* findFrameWithEarliestPts();
  
#if CUSTOM_NVDEC_DEBUG
  // Debug helper functions
  void printPtsQueue(const std::string& context) const;
  void printFrameBuffer(const std::string& context) const;
#endif


  // Initialize video parser
  void createVideoParser();

  // Convert CUDA frame pointer to AVFrame
  UniqueAVFrame convertCudaFrameToAVFrame(
      CUdeviceptr framePtr,
      unsigned int pitch,
      const CUVIDPARSERDISPINFO& dispInfo,
      const AVRational& timeBase);
};

} // namespace facebook::torchcodec
