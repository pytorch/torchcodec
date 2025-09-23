// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <npp.h>
#include "src/torchcodec/_core/DeviceInterface.h"
#include "src/torchcodec/_core/FilterGraph.h"

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
  std::unique_ptr<FiltersContext> initializeFiltersContext(
      const VideoStreamOptions& videoStreamOptions,
      const UniqueAVFrame& avFrame,
      const AVRational& timeBase);

  UniqueAVBufferRef ctx_;
  std::unique_ptr<NppStreamContext> nppCtx_;
  // Current filter context. Used to know whether a new FilterGraph
  // should be created to process the next frame.
  std::unique_ptr<FiltersContext> filtersContext_;
  std::unique_ptr<FilterGraph> filterGraph_;
};

} // namespace facebook::torchcodec
