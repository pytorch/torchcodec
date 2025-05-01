// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/torchcodec/_core/Frame.h"

namespace facebook::torchcodec {

torch::Tensor allocateEmptyHWCTensor(
    int height,
    int width,
    torch::Device device,
    std::optional<int> numFrames) {
  auto tensorOptions = torch::TensorOptions()
                           .dtype(torch::kUInt8)
                           .layout(torch::kStrided)
                           .device(device);
  TORCH_CHECK(height > 0, "height must be > 0, got: ", height);
  TORCH_CHECK(width > 0, "width must be > 0, got: ", width);
  if (numFrames.has_value()) {
    auto numFramesValue = numFrames.value();
    TORCH_CHECK(
        numFramesValue >= 0, "numFrames must be >= 0, got: ", numFramesValue);
    return torch::empty({numFramesValue, height, width, 3}, tensorOptions);
  } else {
    return torch::empty({height, width, 3}, tensorOptions);
  }
}

} // namespace facebook::torchcodec
