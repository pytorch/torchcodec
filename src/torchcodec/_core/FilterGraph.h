// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "src/torchcodec/_core/FFMPEGCommon.h"
#include "src/torchcodec/_core/StreamOptions.h"

namespace facebook::torchcodec {

struct FiltersContext {
  int inputWidth = 0;
  int inputHeight = 0;
  AVPixelFormat inputFormat = AV_PIX_FMT_NONE;
  AVRational inputAspectRatio = {0, 0};
  int outputWidth = 0;
  int outputHeight = 0;
  AVPixelFormat outputFormat = AV_PIX_FMT_NONE;
  std::string filtergraphStr;
  AVRational timeBase = {0, 0};
  UniqueAVBufferRef hwFramesCtx;

  FiltersContext() = default;
  FiltersContext(FiltersContext&&) = default;
  FiltersContext& operator=(FiltersContext&&) = default;
  FiltersContext(
      int inputWidth,
      int inputHeight,
      AVPixelFormat inputFormat,
      AVRational inputAspectRatio,
      int outputWidth,
      int outputHeight,
      AVPixelFormat outputFormat,
      const std::string& filtergraphStr,
      AVRational timeBase,
      AVBufferRef* hwFramesCtx = nullptr);

  bool operator==(const FiltersContext&) const;
  bool operator!=(const FiltersContext&) const;
};

class FilterGraph {
 public:
  FilterGraph(
      const FiltersContext& filtersContext,
      const VideoStreamOptions& videoStreamOptions);

  UniqueAVFrame convert(const UniqueAVFrame& avFrame);

 private:
  UniqueAVFilterGraph filterGraph_;
  AVFilterContext* sourceContext_ = nullptr;
  AVFilterContext* sinkContext_ = nullptr;
};

} // namespace facebook::torchcodec
