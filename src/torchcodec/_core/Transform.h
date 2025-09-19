// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <optional>
#include <string>
#include "src/torchcodec/_core/Frame.h"

namespace facebook::torchcodec {

class Transform {
 public:
  virtual std::string getFilterGraphCpu() const = 0;
  virtual ~Transform() = default;

  // If the transformation does not change the output frame dimensions, then
  // there is no need to override this member function. The default
  // implementation returns an empty optional, indicating that the output frame
  // has the same dimensions as the input frame.
  //
  // If the transformation does change the output frame dimensions, then it
  // must override this member function and return the output frame dimensions.
  virtual std::optional<FrameDims> getOutputFrameDims() const {
    return std::nullopt;
  }

  virtual bool isSwScaleCompatible() const {
    return false;
  }
};

class ResizeTransform : public Transform {
 public:
  enum class InterpolationMode { BILINEAR, BICUBIC, NEAREST };

  ResizeTransform(int width, int height)
      : width_(width),
        height_(height),
        interpolationMode_(InterpolationMode::BILINEAR) {}

  ResizeTransform(int width, int height, InterpolationMode interpolationMode)
      : width_(width), height_(height), interpolationMode_(interpolationMode) {}

  std::string getFilterGraphCpu() const override;
  std::optional<FrameDims> getOutputFrameDims() const override;
  bool isSwScaleCompatible() const override;

  int getSwsFlags() const;

 private:
  int width_;
  int height_;
  InterpolationMode interpolationMode_;
};

} // namespace facebook::torchcodec
