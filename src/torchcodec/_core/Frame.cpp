// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/torchcodec/_core/Frame.h"

namespace facebook::torchcodec {

FrameBatchOutput::FrameBatchOutput(
    int64_t numFrames,
    const FrameDims& outputDims,
    const torch::Device& device)
    : ptsSeconds(torch::empty({numFrames}, {torch::kFloat64})),
      durationSeconds(torch::empty({numFrames}, {torch::kFloat64})) {
  data = allocateEmptyHWCTensor(outputDims, device, numFrames);
}

torch::Tensor allocateEmptyHWCTensor(
    const FrameDims& frameDims,
    const torch::Device& device,
    std::optional<int> numFrames) {
  auto tensorOptions = torch::TensorOptions()
                           .dtype(torch::kUInt8)
                           .layout(torch::kStrided)
                           .device(device);
  TORCH_CHECK(
      frameDims.height > 0, "height must be > 0, got: ", frameDims.height);
  TORCH_CHECK(frameDims.width > 0, "width must be > 0, got: ", frameDims.width);
  if (numFrames.has_value()) {
    auto numFramesValue = numFrames.value();
    TORCH_CHECK(
        numFramesValue >= 0, "numFrames must be >= 0, got: ", numFramesValue);
    return torch::empty(
        {numFramesValue, frameDims.height, frameDims.width, 3}, tensorOptions);
  } else {
    return torch::empty({frameDims.height, frameDims.width, 3}, tensorOptions);
  }
}

} // namespace facebook::torchcodec
