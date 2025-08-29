// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "src/torchcodec/_core/DeviceInterface.h"
#include "src/torchcodec/_core/FFMPEGCommon.h"
#include "src/torchcodec/_core/FilterGraph.h"

namespace facebook::torchcodec {

class CpuDeviceInterface : public DeviceInterface {
 public:
  CpuDeviceInterface(const torch::Device& device);

  virtual ~CpuDeviceInterface() {}

  std::optional<const AVCodec*> findCodec(
      [[maybe_unused]] const AVCodecID& codecId) override {
    return std::nullopt;
  }

  void initializeContext(
      [[maybe_unused]] AVCodecContext* codecContext) override {}

  void convertAVFrameToFrameOutput(
      const VideoStreamOptions& videoStreamOptions,
      const AVRational& timeBase,
      UniqueAVFrame& avFrame,
      FrameOutput& frameOutput,
      std::optional<torch::Tensor> preAllocatedOutputTensor =
          std::nullopt) override;

 private:
  int convertAVFrameToTensorUsingSwsScale(
      const UniqueAVFrame& avFrame,
      torch::Tensor& outputTensor);

  torch::Tensor convertAVFrameToTensorUsingFilterGraph(
      const UniqueAVFrame& avFrame);

  struct SwsFrameContext {
    int inputWidth;
    int inputHeight;
    AVPixelFormat inputFormat;
    int outputWidth;
    int outputHeight;
    bool operator==(const SwsFrameContext&) const;
    bool operator!=(const SwsFrameContext&) const;
  };

  void createSwsContext(
      const SwsFrameContext& swsFrameContext,
      const enum AVColorSpace colorspace);

  // color-conversion fields. Only one of FilterGraphContext and
  // UniqueSwsContext should be non-null.
  std::unique_ptr<FilterGraph> filterGraphContext_;
  UniqueSwsContext swsContext_;

  // Used to know whether a new FilterGraphContext or UniqueSwsContext should
  // be created before decoding a new frame.
  SwsFrameContext prevSwsFrameContext_;
  FiltersContext prevFiltersContext_;
};

} // namespace facebook::torchcodec
