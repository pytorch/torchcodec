// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <map>
#include <memory>
#include <mutex>

#include <cuda.h>

#include "NVCUVIDRuntimeLoader.h"
#include "NVDECCacheConfig.h"
#include "StableABICompat.h"
#include "nvcuvid_include/cuviddec.h"
#include "nvcuvid_include/nvcuvid.h"

namespace facebook::torchcodec {

// This file implements a cache for NVDEC decoders.
// TODONVDEC P3: Consider merging this with Cache.h. The main difference is that
// this NVDEC Cache involves a cache key (the decoder parameters).

struct CUvideoDecoderDeleter {
  void operator()(CUvideodecoder* decoder_ptr) const {
    if (decoder_ptr && *decoder_ptr) {
      cuvidDestroyDecoder(*decoder_ptr);
      delete decoder_ptr;
    }
  }
};

using UniqueCUvideodecoder =
    std::unique_ptr<CUvideodecoder, CUvideoDecoderDeleter>;

struct CacheEntry {
  UniqueCUvideodecoder decoder;
  uint64_t last_used; // LRU timestamp

  CacheEntry(UniqueCUvideodecoder dec, uint64_t ts)
      : decoder(std::move(dec)), last_used(ts) {}
};

// A per-device LRU cache for NVDEC decoders. There is one instance of this
// class per GPU device, and it is accessed through the static getCache()
// method.  The cache supports multiple decoders with the same parameters.
class NVDECCache {
 public:
  static NVDECCache& get_cache(const StableDevice& device);

  // Get decoder from cache - returns nullptr if none available.
  UniqueCUvideodecoder get_decoder(
      CUVIDEOFORMAT* video_format,
      cudaVideoSurfaceFormat surface_format);

  // Return decoder to cache using LRU eviction.
  void return_decoder(
      CUVIDEOFORMAT* video_format,
      cudaVideoSurfaceFormat surface_format,
      UniqueCUvideodecoder decoder);

  // Iterates all per-device cache instances and evicts LRU entries until each
  // cache's size is at most capacity. Called from setNVDECCacheCapacity().
  static void evict_excess_entries_across_devices(int capacity);

  // Returns the number of entries in the cache for a given device index.
  static int get_cache_size_for_device(int device_index);

 private:
  // Cache key struct: a decoder can be reused and taken from the cache only if
  // all these parameters match.
  struct CacheKey {
    cudaVideoCodec codec_type;
    uint32_t width;
    uint32_t height;
    // The display area is the crop that NVDEC applies to the coded frame, and
    // it is baked into the decoder when it is created: it dictates the
    // dimensions of the output surface. Different videos can have the same
    // coded dimensions but a different display area (heights of 530 and 532 are
    // both coded as 544), so it has to be part of the key. Reusing a decoder
    // whose output surface isn't the size we expect silently corrupts frames.
    int display_left;
    int display_top;
    int display_right;
    int display_bottom;
    cudaVideoChromaFormat chroma_format;
    uint32_t bit_depth_luma_minus8;
    uint8_t num_decode_surfaces;
    cudaVideoSurfaceFormat output_surface_format;

    CacheKey() = delete;

    explicit CacheKey(
        CUVIDEOFORMAT* video_format,
        cudaVideoSurfaceFormat surface_fmt) {
      STD_TORCH_CHECK(video_format != nullptr, "videoFormat must not be null");
      codec_type = video_format->codec;
      width = video_format->coded_width;
      height = video_format->coded_height;
      display_left = video_format->display_area.left;
      display_top = video_format->display_area.top;
      display_right = video_format->display_area.right;
      display_bottom = video_format->display_area.bottom;
      chroma_format = video_format->chroma_format;
      bit_depth_luma_minus8 = video_format->bit_depth_luma_minus8;
      num_decode_surfaces = video_format->min_num_decode_surfaces;
      output_surface_format = surface_fmt;
    }

    CacheKey(const CacheKey&) = default;
    CacheKey& operator=(const CacheKey&) = default;

    bool operator<(const CacheKey& other) const {
      return std::tie(
                 codec_type,
                 width,
                 height,
                 display_left,
                 display_top,
                 display_right,
                 display_bottom,
                 chroma_format,
                 bit_depth_luma_minus8,
                 num_decode_surfaces,
                 output_surface_format) <
          std::tie(
                 other.codec_type,
                 other.width,
                 other.height,
                 other.display_left,
                 other.display_top,
                 other.display_right,
                 other.display_bottom,
                 other.chroma_format,
                 other.bit_depth_luma_minus8,
                 other.num_decode_surfaces,
                 other.output_surface_format);
    }
  };

  NVDECCache() = default;
  ~NVDECCache() = default;

  void evict_lru_entry();

  static NVDECCache* get_cache_instances();

  std::multimap<CacheKey, CacheEntry> cache_;
  std::mutex cache_lock_;
  uint64_t last_used_counter_ = 0;
};

} // namespace facebook::torchcodec
