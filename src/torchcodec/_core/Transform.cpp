// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/torchcodec/_core/Transform.h"
#include <torch/types.h>
#include "src/torchcodec/_core/FFMPEGCommon.h"

namespace facebook::torchcodec {

namespace {

std::string toFilterGraphInterpolation(
    ResizeTransform::InterpolationMode mode) {
  switch (mode) {
    case ResizeTransform::InterpolationMode::BILINEAR:
      return "bilinear";
    case ResizeTransform::InterpolationMode::BICUBIC:
      return "bicubic";
    case ResizeTransform::InterpolationMode::NEAREST:
      return "nearest";
    default:
      TORCH_CHECK(
          false,
          "Unknown interpolation mode: " +
              std::to_string(static_cast<int>(mode)));
  }
}

int toSwsInterpolation(ResizeTransform::InterpolationMode mode) {
  switch (mode) {
    case ResizeTransform::InterpolationMode::BILINEAR:
      return SWS_BILINEAR;
    case ResizeTransform::InterpolationMode::BICUBIC:
      return SWS_BICUBIC;
    case ResizeTransform::InterpolationMode::NEAREST:
      return SWS_POINT;
    default:
      TORCH_CHECK(
          false,
          "Unknown interpolation mode: " +
              std::to_string(static_cast<int>(mode)));
  }
}

} // namespace

std::string ResizeTransform::getFilterGraphCpu() const {
  return "scale=" + std::to_string(width_) + ":" + std::to_string(height_) +
      ":sws_flags=" + toFilterGraphInterpolation(interpolationMode_);
}

std::optional<FrameDims> ResizeTransform::getOutputFrameDims() const {
  return FrameDims(width_, height_);
}

bool ResizeTransform::isSwScaleCompatible() const {
  return true;
}

int ResizeTransform::getSwsFlags() const {
  return toSwsInterpolation(interpolationMode_);
}

} // namespace facebook::torchcodec
