// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/torchcodec/_core/Transform.h"
#include <torch/types.h>

namespace facebook::torchcodec {

std::string toStringFilterGraph(Transform::InterpolationMode mode) {
  switch (mode) {
    case Transform::InterpolationMode::BILINEAR:
      return "BILINEAR";
    case Transform::InterpolationMode::BICUBIC:
      return "BICUBIC";
    case Transform::InterpolationMode::NEAREST:
      return "NEAREST";
    default:
      TORCH_CHECK(false, "Unknown interpolation mode: " + std::to_string(mode));
  }
}

std::string Transform::getFilterGraphCpu() const {
  return "scale=width=" + std::to_string(width_) +
      ":height=" + std::to_string(height_) +
      ":sws_flags=" + toStringFilterGraph(interpolationMode_);
}

} // namespace facebook::torchcodec
