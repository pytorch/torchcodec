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

  std::unique_ptr<FiltersContext> initializeFiltersContext(
      const VideoStreamOptions& videoStreamOptions,
      const UniqueAVFrame& avFrame,
      const AVRational& timeBase) override;

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

  std::unique_ptr<FiltersContext> initializeFiltersContextInternal(
      const VideoStreamOptions& videoStreamOptions,
      const UniqueAVFrame& avFrame,
      const AVRational& timeBase);

  void createSwsContext(
      const FiltersContext& filtersContext,
      const enum AVColorSpace colorspace);

  // SWS color conversion context
  UniqueSwsContext swsContext_;

  // Used to know whether a new UniqueSwsContext should
  // be created before decoding a new frame.
  std::unique_ptr<FiltersContext> prevFiltersContext_;
};

} // namespace facebook::torchcodec
