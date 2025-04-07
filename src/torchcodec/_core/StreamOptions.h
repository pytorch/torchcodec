// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/types.h>
#include <optional>
#include <string>

namespace facebook::torchcodec {

enum ColorConversionLibrary {
  // TODO: Add an AUTO option later.
  // Use the libavfilter library for color conversion.
  FILTERGRAPH,
  // Use the libswscale library for color conversion.
  SWSCALE
};

struct VideoStreamOptions {
  VideoStreamOptions() {}

  // Number of threads we pass to FFMPEG for decoding.
  // 0 means FFMPEG will choose the number of threads automatically to fully
  // utilize all cores. If not set, it will be the default FFMPEG behavior for
  // the given codec.
  std::optional<int> ffmpegThreadCount;
  // Currently the dimension order can be either NHWC or NCHW.
  // H=height, W=width, C=channel.
  std::string dimensionOrder = "NCHW";
  // The output height and width of the frame. If not specified, the output
  // is the same as the original video.
  std::optional<int> width;
  std::optional<int> height;
  std::optional<ColorConversionLibrary> colorConversionLibrary;
  // By default we use CPU for decoding for both C++ and python users.
  torch::Device device = torch::kCPU;
};

struct AudioStreamOptions {
  AudioStreamOptions() {}

  std::optional<int> sampleRate;
};

} // namespace facebook::torchcodec
