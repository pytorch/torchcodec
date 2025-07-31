// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <npp.h>
#include "src/torchcodec/_core/DeviceInterface.h"

namespace facebook::torchcodec {

class CudaDeviceInterface : public DeviceInterface {
 public:
  CudaDeviceInterface(const torch::Device& device);

  virtual ~CudaDeviceInterface();

  std::optional<const AVCodec*> findCodec(const AVCodecID& codecId) override;

  void initializeContext(AVCodecContext* codecContext) override;

  void convertAVFrameToFrameOutput(
      const VideoStreamOptions& videoStreamOptions,
      const AVRational& timeBase,
      UniqueAVFrame& avFrame,
      FrameOutput& frameOutput,
      std::optional<torch::Tensor> preAllocatedOutputTensor =
          std::nullopt) override;

 private:
  AVBufferRef* ctx_ = nullptr;

  void colorConvert8bitFrame(
      UniqueAVFrame& avFrame,
      torch::Tensor& dst,
      const NppiSize& oSizeROI,
      const NppStreamContext& nppCtx);

  void colorConvert10bitFrame(
      UniqueAVFrame& avFrame,
      torch::Tensor& dst,
      const NppiSize& oSizeROI,
      const NppStreamContext& nppCtx);
};

} // namespace facebook::torchcodec
