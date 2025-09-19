// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/torchcodec/_core/CpuDeviceInterface.h"

namespace facebook::torchcodec {
namespace {

static bool g_cpu = registerDeviceInterface(
    torch::kCPU,
    [](const torch::Device& device) { return new CpuDeviceInterface(device); });

} // namespace

CpuDeviceInterface::SwsFrameContext::SwsFrameContext(
    int inputWidth,
    int inputHeight,
    AVPixelFormat inputFormat,
    int outputWidth,
    int outputHeight)
    : inputWidth(inputWidth),
      inputHeight(inputHeight),
      inputFormat(inputFormat),
      outputWidth(outputWidth),
      outputHeight(outputHeight) {}

bool CpuDeviceInterface::SwsFrameContext::operator==(
    const CpuDeviceInterface::SwsFrameContext& other) const {
  return inputWidth == other.inputWidth && inputHeight == other.inputHeight &&
      inputFormat == other.inputFormat && outputWidth == other.outputWidth &&
      outputHeight == other.outputHeight;
}

bool CpuDeviceInterface::SwsFrameContext::operator!=(
    const CpuDeviceInterface::SwsFrameContext& other) const {
  return !(*this == other);
}

CpuDeviceInterface::CpuDeviceInterface(const torch::Device& device)
    : DeviceInterface(device) {
  TORCH_CHECK(g_cpu, "CpuDeviceInterface was not registered!");
  TORCH_CHECK(
      device_.type() == torch::kCPU, "Unsupported device: ", device_.str());
}

void CpuDeviceInterface::initialize(
    [[maybe_unused]] AVCodecContext* codecContext,
    const VideoStreamOptions& videoStreamOptions,
    const std::vector<std::unique_ptr<Transform>>& transforms,
    const AVRational& timeBase,
    const FrameDims& outputDims) {
  videoStreamOptions_ = videoStreamOptions;
  timeBase_ = timeBase;
  outputDims_ = outputDims;

  // TODO: rationalize comment below with new stuff.
  // By default, we want to use swscale for color conversion because it is
  // faster. However, it has width requirements, so we may need to fall back
  // to filtergraph. We also need to respect what was requested from the
  // options; we respect the options unconditionally, so it's possible for
  // swscale's width requirements to be violated. We don't expose the ability to
  // choose color conversion library publicly; we only use this ability
  // internally.

  // We can only use swscale when we have a single resize transform. Note that
  // this means swscale will not support the case of having several,
  // back-to-base resizes. There's no strong reason to even do that, but if
  // someone does, it's more correct to implement that with filtergraph.
  bool areTransformsSwScaleCompatible = transforms.empty() ||
      (transforms.size() == 1 && transforms[0]->isResize());

  // swscale requires widths to be multiples of 32:
  // https://stackoverflow.com/questions/74351955/turn-off-sw-scale-conversion-to-planar-yuv-32-byte-alignment-requirements
  bool isWidthSwScaleCompatible = (outputDims_.width % 32) == 0;

  bool userRequestedSwScale =
      videoStreamOptions_.colorConversionLibrary.has_value() &&
      videoStreamOptions_.colorConversionLibrary.value() ==
          ColorConversionLibrary::SWSCALE;

  // Note that we treat the transform limitation differently from the width
  // limitation. That is, we consider the transforms being compatible with
  // sws_scale as a hard requirement. If the transforms are not compatiable,
  // then we will end up not applying the transforms, and that is wrong.
  //
  // The width requirement, however, is a soft requirement. Even if we don't
  // meet it, we let the user override it. We have tests that depend on this
  // behavior. Since we don't expose the ability to choose swscale or
  // filtergraph in our public API, this is probably okay. It's also the only
  // way that we can be certain we are testing one versus the other.
  if (areTransformsSwScaleCompatible &&
      (userRequestedSwScale || isWidthSwScaleCompatible)) {
    colorConversionLibrary_ = ColorConversionLibrary::SWSCALE;

    // We established above that if the transforms are swscale compatible and
    // non-empty, then they must have only one transforms, and that transform is
    // ResizeTransform.
    if (!transforms.empty()) {
      auto resize = dynamic_cast<ResizeTransform*>(transforms[0].get());
      TORCH_CHECK(resize != nullptr, "ResizeTransform expected but not found!")
      swsFlags_ = resize->getSwsFlags();
    }
  } else {
    colorConversionLibrary_ = ColorConversionLibrary::FILTERGRAPH;

    // If we have any transforms, replace filters_ with the filter strings from
    // the transforms.
    std::stringstream filters;
    bool first = true;
    for (const auto& transform : transforms) {
      if (!first) {
        filters << ",";
      }
      filters << transform->getFilterGraphCpu();
      first = false;
    }
    if (!transforms.empty()) {
      filters_ = filters.str();
    }
  }

  initialized_ = true;
}

// Note [preAllocatedOutputTensor with swscale and filtergraph]:
// Callers may pass a pre-allocated tensor, where the output.data tensor will
// be stored. This parameter is honored in any case, but it only leads to a
// speed-up when swscale is used. With swscale, we can tell ffmpeg to place the
// decoded frame directly into `preAllocatedtensor.data_ptr()`. We haven't yet
// found a way to do that with filtegraph.
// TODO: Figure out whether that's possible!
// Dimension order of the preAllocatedOutputTensor must be HWC, regardless of
// `dimension_order` parameter. It's up to callers to re-shape it if needed.
void CpuDeviceInterface::convertAVFrameToFrameOutput(
    UniqueAVFrame& avFrame,
    FrameOutput& frameOutput,
    std::optional<torch::Tensor> preAllocatedOutputTensor) {
  TORCH_CHECK(initialized_, "CpuDeviceInterface was not initialized.");
  if (preAllocatedOutputTensor.has_value()) {
    auto shape = preAllocatedOutputTensor.value().sizes();
    TORCH_CHECK(
        (shape.size() == 3) && (shape[0] == outputDims_.height) &&
            (shape[1] == outputDims_.width) && (shape[2] == 3),
        "Expected pre-allocated tensor of shape ",
        outputDims_.height,
        "x",
        outputDims_.width,
        "x3, got ",
        shape);
  }

  torch::Tensor outputTensor;
  enum AVPixelFormat frameFormat =
      static_cast<enum AVPixelFormat>(avFrame->format);

  if (colorConversionLibrary_ == ColorConversionLibrary::SWSCALE) {
    // We need to compare the current frame context with our previous frame
    // context. If they are different, then we need to re-create our colorspace
    // conversion objects. We create our colorspace conversion objects late so
    // that we don't have to depend on the unreliable metadata in the header.
    // And we sometimes re-create them because it's possible for frame
    // resolution to change mid-stream. Finally, we want to reuse the colorspace
    // conversion objects as much as possible for performance reasons.
    SwsFrameContext swsFrameContext(
        avFrame->width,
        avFrame->height,
        frameFormat,
        outputDims_.width,
        outputDims_.height);

    outputTensor = preAllocatedOutputTensor.value_or(
        allocateEmptyHWCTensor(outputDims_, torch::kCPU));

    if (!swsContext_ || prevSwsFrameContext_ != swsFrameContext) {
      createSwsContext(swsFrameContext, avFrame->colorspace);
      prevSwsFrameContext_ = swsFrameContext;
    }
    int resultHeight =
        convertAVFrameToTensorUsingSwScale(avFrame, outputTensor);
    // If this check failed, it would mean that the frame wasn't reshaped to
    // the expected height.
    // TODO: Can we do the same check for width?
    TORCH_CHECK(
        resultHeight == outputDims_.height,
        "resultHeight != outputDims_.height: ",
        resultHeight,
        " != ",
        outputDims_.height);

    frameOutput.data = outputTensor;
  } else if (colorConversionLibrary_ == ColorConversionLibrary::FILTERGRAPH) {
    FiltersContext filtersContext(
        avFrame->width,
        avFrame->height,
        frameFormat,
        avFrame->sample_aspect_ratio,
        outputDims_.width,
        outputDims_.height,
        AV_PIX_FMT_RGB24,
        filters_,
        timeBase_);

    if (!filterGraphContext_ || prevFiltersContext_ != filtersContext) {
      filterGraphContext_ =
          std::make_unique<FilterGraph>(filtersContext, videoStreamOptions_);
      prevFiltersContext_ = std::move(filtersContext);
    }
    outputTensor = convertAVFrameToTensorUsingFilterGraph(avFrame);

    // Similarly to above, if this check fails it means the frame wasn't
    // reshaped to its expected dimensions by filtergraph.
    auto shape = outputTensor.sizes();
    TORCH_CHECK(
        (shape.size() == 3) && (shape[0] == outputDims_.height) &&
            (shape[1] == outputDims_.width) && (shape[2] == 3),
        "Expected output tensor of shape ",
        outputDims_.height,
        "x",
        outputDims_.width,
        "x3, got ",
        shape);

    if (preAllocatedOutputTensor.has_value()) {
      // We have already validated that preAllocatedOutputTensor and
      // outputTensor have the same shape.
      preAllocatedOutputTensor.value().copy_(outputTensor);
      frameOutput.data = preAllocatedOutputTensor.value();
    } else {
      frameOutput.data = outputTensor;
    }
  } else {
    TORCH_CHECK(
        false,
        "Invalid color conversion library: ",
        static_cast<int>(colorConversionLibrary_));
  }
}

int CpuDeviceInterface::convertAVFrameToTensorUsingSwScale(
    const UniqueAVFrame& avFrame,
    torch::Tensor& outputTensor) {
  uint8_t* pointers[4] = {
      outputTensor.data_ptr<uint8_t>(), nullptr, nullptr, nullptr};
  int expectedOutputWidth = outputTensor.sizes()[1];
  int linesizes[4] = {expectedOutputWidth * 3, 0, 0, 0};
  int resultHeight = sws_scale(
      swsContext_.get(),
      avFrame->data,
      avFrame->linesize,
      0,
      avFrame->height,
      pointers,
      linesizes);
  return resultHeight;
}

torch::Tensor CpuDeviceInterface::convertAVFrameToTensorUsingFilterGraph(
    const UniqueAVFrame& avFrame) {
  UniqueAVFrame filteredAVFrame = filterGraphContext_->convert(avFrame);

  TORCH_CHECK_EQ(filteredAVFrame->format, AV_PIX_FMT_RGB24);

  int height = filteredAVFrame->height;
  int width = filteredAVFrame->width;
  std::vector<int64_t> shape = {height, width, 3};
  std::vector<int64_t> strides = {filteredAVFrame->linesize[0], 3, 1};
  AVFrame* filteredAVFramePtr = filteredAVFrame.release();
  auto deleter = [filteredAVFramePtr](void*) {
    UniqueAVFrame avFrameToDelete(filteredAVFramePtr);
  };
  return torch::from_blob(
      filteredAVFramePtr->data[0], shape, strides, deleter, {torch::kUInt8});
}

void CpuDeviceInterface::createSwsContext(
    const SwsFrameContext& swsFrameContext,
    const enum AVColorSpace colorspace) {
  SwsContext* swsContext = sws_getContext(
      swsFrameContext.inputWidth,
      swsFrameContext.inputHeight,
      swsFrameContext.inputFormat,
      swsFrameContext.outputWidth,
      swsFrameContext.outputHeight,
      AV_PIX_FMT_RGB24,
      SWS_BILINEAR,
      nullptr,
      nullptr,
      nullptr);
  TORCH_CHECK(swsContext, "sws_getContext() returned nullptr");

  int* invTable = nullptr;
  int* table = nullptr;
  int srcRange, dstRange, brightness, contrast, saturation;
  int ret = sws_getColorspaceDetails(
      swsContext,
      &invTable,
      &srcRange,
      &table,
      &dstRange,
      &brightness,
      &contrast,
      &saturation);
  TORCH_CHECK(ret != -1, "sws_getColorspaceDetails returned -1");

  const int* colorspaceTable = sws_getCoefficients(colorspace);
  ret = sws_setColorspaceDetails(
      swsContext,
      colorspaceTable,
      srcRange,
      colorspaceTable,
      dstRange,
      brightness,
      contrast,
      saturation);
  TORCH_CHECK(ret != -1, "sws_setColorspaceDetails returned -1");

  swsContext_.reset(swsContext);
}

} // namespace facebook::torchcodec
