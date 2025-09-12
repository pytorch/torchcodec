// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <string>

namespace facebook::torchcodec {

class Transform {
 public:
  std::string getFilterGraphCpu() const = 0
};

class ResizeTransform : public Transform {
 public:
  ResizeTransform(int width, int height)
      : width_(width),
        height_(height),
        interpolation_(InterpolationMode::BILINEAR) {}

  ResizeTransform(int width, int height, InterpolationMode interpolation) =
      default;

  std::string getFilterGraphCpu() const override;

  enum class InterpolationMode { BILINEAR, BICUBIC, NEAREST };

 private:
  int width_;
  int height_;
  InterpolationMode interpolation_;
}

} // namespace facebook::torchcodec
