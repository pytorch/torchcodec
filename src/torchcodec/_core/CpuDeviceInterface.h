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

  virtual void initialize(
      [[maybe_unused]] AVCodecContext* codecContext,
      const VideoStreamOptions& videoStreamOptions,
      const std::vector<std::unique_ptr<Transform>>& transforms,
      const AVRational& timeBase,
      [[maybe_unused]] const FrameDims& metadataDims,
      const std::optional<FrameDims>& resizedOutputDims) override;

  void convertAVFrameToFrameOutput(
      UniqueAVFrame& avFrame,
      FrameOutput& frameOutput,
      std::optional<torch::Tensor> preAllocatedOutputTensor =
          std::nullopt) override;

 private:
  int convertAVFrameToTensorUsingSwScale(
      const UniqueAVFrame& avFrame,
      torch::Tensor& outputTensor);

  ColorConversionLibrary getColorConversionLibrary(
      const FrameDims& inputFrameDims);

  struct SwsFrameContext {
    int inputWidth = 0;
    int inputHeight = 0;
    AVPixelFormat inputFormat = AV_PIX_FMT_NONE;
    int outputWidth = 0;
    int outputHeight = 0;

    SwsFrameContext() = default;
    SwsFrameContext(
        int inputWidth,
        int inputHeight,
        AVPixelFormat inputFormat,
        int outputWidth,
        int outputHeight);
    bool operator==(const SwsFrameContext&) const;
    bool operator!=(const SwsFrameContext&) const;
  };

  void createSwsContext(
      const SwsFrameContext& swsFrameContext,
      const enum AVColorSpace colorspace);

  VideoStreamOptions videoStreamOptions_;
  AVRational timeBase_;
  std::optional<FrameDims> resizedOutputDims_;

  // Color-conversion objects. Only one of filterGraph_ and swsContext_ should
  // be non-null. Which one we use is controlled by colorConversionLibrary_.
  //
  // Creating both filterGraph_ and swsContext_ is relatively expensive, so we
  // reuse them across frames. However, it is possbile that subsequent frames
  // are different enough (change in dimensions) that we can't reuse the color
  // conversion object. We store the relevant frame context from the frame used
  // to create the object last time. We always compare the current frame's info
  // against the previous one to determine if we need to recreate the color
  // conversion object.
  //
  // TODO: The names of these fields is confusing, as the actual color
  //       conversion object for Sws has "context" in the name,  and we use
  //       "context" for the structs we store to know if we need to recreate a
  //       color conversion object. We should clean that up.
  std::unique_ptr<FilterGraph> filterGraph_;
  FiltersContext prevFiltersContext_;
  UniqueSwsContext swsContext_;
  SwsFrameContext prevSwsFrameContext_;

  // The filter we supply to filterGraph_, if it is used. The copy filter just
  // copies the input to the output. Computationally, it should be a no-op. If
  // we get no user-provided transforms, we will use the copy filter. Otherwise,
  // we will construct the string from the transforms.
  std::string filters_ = "copy";

  // The flags we supply to swsContext_, if it used. The flags control the
  // resizing algorithm. We default to bilinear. Users can override this with a
  // ResizeTransform.
  int swsFlags_ = SWS_BILINEAR;

  // Values set during initialization and referred to in
  // getColorConversionLibrary().
  bool areTransformsSwScaleCompatible_;
  bool userRequestedSwScale_;

  bool initialized_ = false;
};

} // namespace facebook::torchcodec
