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

  void initialize(
      AVCodecContext* codecContext,
      const VideoStreamOptions& videoStreamOptions,
      [[maybe_unused]] const std::vector<std::unique_ptr<Transform>>&
          transforms,
      const AVRational& timeBase,
      const FrameDims& outputDims) override;

  void convertAVFrameToFrameOutput(
      UniqueAVFrame& avFrame,
      FrameOutput& frameOutput,
      std::optional<torch::Tensor> preAllocatedOutputTensor =
          std::nullopt) override;

 private:
  // Our CUDA decoding code assumes NV12 format. In order to handle other
  // kindsof input, we need to convert them to NV12. Our current implementation
  // does this using filtergraph.
  UniqueAVFrame maybeConvertAVFrameToNV12(UniqueAVFrame& avFrame);

  VideoStreamOptions videoStreamOptions_;
  AVRational timeBase_;
  FrameDims outputDims_;

  UniqueAVBufferRef ctx_;
  std::unique_ptr<NppStreamContext> nppCtx_;

  // This filtergraph instance is only used for NV12 format conversion in
  // maybeConvertAVFrameToNV12().
  std::unique_ptr<FiltersContext> nv12ConversionContext_;
  std::unique_ptr<FilterGraph> nv12Conversion_;
};

} // namespace facebook::torchcodec
