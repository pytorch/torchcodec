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

  // By default we have to use filtergraph, as it is more general. We can only
  // use swscale when we have met strict requirements. See
  // CpuDeviceInterface::initialze() for the logic.
  ColorConversionLibrary colorConversionLibrary =
      ColorConversionLibrary::FILTERGRAPH;

  // By default we use CPU for decoding for both C++ and python users.
  torch::Device device = torch::kCPU;

  // Encoding options
  std::optional<int> bitRate;
  std::optional<int> gopSize;
  std::optional<int> maxBFrames;
};

struct AudioStreamOptions {
  AudioStreamOptions() {}

  // Encoding only
  std::optional<int> bitRate;
  // Decoding and encoding:
  std::optional<int> numChannels;
  std::optional<int> sampleRate;
};

} // namespace facebook::torchcodec
