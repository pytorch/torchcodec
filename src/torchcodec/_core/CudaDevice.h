// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "src/torchcodec/_core/DeviceInterface.h"

namespace facebook::torchcodec {

class CudaDevice : public DeviceInterface {
 public:
  CudaDevice(const torch::Device& device);

  virtual ~CudaDevice();

  std::optional<const AVCodec*> findCodec(const AVCodecID& codecId) override;

  void initializeContext(AVCodecContext* codecContext) override;

  void convertAVFrameToFrameOutput(
      const SingleStreamDecoder::VideoStreamOptions& videoStreamOptions,
      UniqueAVFrame& avFrame,
      SingleStreamDecoder::FrameOutput& frameOutput,
      std::optional<torch::Tensor> preAllocatedOutputTensor =
          std::nullopt) override;

 private:
  AVBufferRef* ctx_ = nullptr;
};

} // namespace facebook::torchcodec
