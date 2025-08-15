// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "src/torchcodec/_core/FFMPEGCommon.h"
#include "src/torchcodec/_core/StreamOptions.h"

namespace facebook::torchcodec {

struct DecodedFrameContext {
  int decodedWidth;
  int decodedHeight;
  AVPixelFormat decodedFormat;
  AVRational decodedAspectRatio;
  int expectedWidth;
  int expectedHeight;

  bool operator==(const DecodedFrameContext&);
  bool operator!=(const DecodedFrameContext&);
};

class FilterGraph {
 public:
  FilterGraph(
      const DecodedFrameContext& frameContext,
      const VideoStreamOptions& videoStreamOptions,
      const AVRational& timeBase);

  UniqueAVFrame convert(const UniqueAVFrame& avFrame);

 private:
  UniqueAVFilterGraph filterGraph_;
  AVFilterContext* sourceContext_ = nullptr;
  AVFilterContext* sinkContext_ = nullptr;
};

} // namespace facebook::torchcodec
