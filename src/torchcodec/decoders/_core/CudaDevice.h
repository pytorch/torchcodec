// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "src/torchcodec/decoders/_core/DeviceInterface.h"

namespace facebook::torchcodec {

struct CudaDevice : public DeviceInterface {
  CudaDevice(const std::string& device);

  virtual ~CudaDevice(){};

  std::optional<const AVCodec*> findCodec(const AVCodecID& codecId) override;

  void initializeContext(AVCodecContext* codecContext) override;

  void convertAVFrameToFrameOutput(
      const VideoDecoder::VideoStreamOptions& videoStreamOptions,
      UniqueAVFrame& avFrame,
      VideoDecoder::FrameOutput& frameOutput,
      std::optional<torch::Tensor> preAllocatedOutputTensor =
          std::nullopt) override;

  void releaseContext(AVCodecContext* codecContext) override;
};

} // namespace facebook::torchcodec
