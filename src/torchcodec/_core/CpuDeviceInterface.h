// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "src/torchcodec/_core/DeviceInterface.h"

namespace facebook::torchcodec {

class CpuDeviceInterface : public DeviceInterface {
 public:
  CpuDeviceInterface(const torch::Device& device, const AVRational& timeBase);

  virtual ~CpuDeviceInterface() {}

  std::optional<const AVCodec*> findCodec(
      [[maybe_unused]] const AVCodecID& codecId) override {
    return std::nullopt;
  }

  void initializeContext(
      [[maybe_unused]] AVCodecContext* codecContext) override {}

  void convertAVFrameToFrameOutput(
      const VideoStreamOptions& videoStreamOptions,
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

  struct FilterGraphContext {
    UniqueAVFilterGraph filterGraph;
    AVFilterContext* sourceContext = nullptr;
    AVFilterContext* sinkContext = nullptr;
  };

  struct DecodedFrameContext {
    int decodedWidth;
    int decodedHeight;
    AVPixelFormat decodedFormat;
    AVRational decodedAspectRatio;
    int expectedWidth;
    int expectedHeight;
    bool operator==(const DecodedFrameContext&);
    bool operator!=(const DecodedFrameContext&);
  };

  void createSwsContext(
      const DecodedFrameContext& frameContext,
      const enum AVColorSpace colorspace);

  void createFilterGraph(
      const DecodedFrameContext& frameContext,
      const VideoStreamOptions& videoStreamOptions);

  // color-conversion fields. Only one of FilterGraphContext and
  // UniqueSwsContext should be non-null.
  FilterGraphContext filterGraphContext_;
  UniqueSwsContext swsContext_;

  // Used to know whether a new FilterGraphContext or UniqueSwsContext should
  // be created before decoding a new frame.
  DecodedFrameContext prevFrameContext_;
};

} // namespace facebook::torchcodec
