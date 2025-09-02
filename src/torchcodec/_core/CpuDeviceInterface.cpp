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

ColorConversionLibrary getColorConversionLibrary(
    const VideoStreamOptions& videoStreamOptions,
    int width) {
  // By default, we want to use swscale for color conversion because it is
  // faster. However, it has width requirements, so we may need to fall back
  // to filtergraph. We also need to respect what was requested from the
  // options; we respect the options unconditionally, so it's possible for
  // swscale's width requirements to be violated. We don't expose the ability to
  // choose color conversion library publicly; we only use this ability
  // internally.

  // swscale requires widths to be multiples of 32:
  // https://stackoverflow.com/questions/74351955/turn-off-sw-scale-conversion-to-planar-yuv-32-byte-alignment-requirements
  // so we fall back to filtergraph if the width is not a multiple of 32.
  auto defaultLibrary = (width % 32 == 0) ? ColorConversionLibrary::SWSCALE
                                          : ColorConversionLibrary::FILTERGRAPH;

  ColorConversionLibrary colorConversionLibrary =
      videoStreamOptions.colorConversionLibrary.value_or(defaultLibrary);

  TORCH_CHECK(
      colorConversionLibrary == ColorConversionLibrary::SWSCALE ||
          colorConversionLibrary == ColorConversionLibrary::FILTERGRAPH,
      "Invalid color conversion library: ",
      static_cast<int>(colorConversionLibrary));
  return colorConversionLibrary;
}

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

std::unique_ptr<FiltersContext> CpuDeviceInterface::initializeFiltersContext(
    const VideoStreamOptions& videoStreamOptions,
    const UniqueAVFrame& avFrame,
    const AVRational& timeBase) {
  enum AVPixelFormat frameFormat =
      static_cast<enum AVPixelFormat>(avFrame->format);
  auto frameDims =
      getHeightAndWidthFromOptionsOrAVFrame(videoStreamOptions, avFrame);
  int expectedOutputHeight = frameDims.height;
  int expectedOutputWidth = frameDims.width;

  if (getColorConversionLibrary(videoStreamOptions, expectedOutputWidth) ==
      ColorConversionLibrary::SWSCALE) {
    return nullptr;
  }

  std::stringstream filters;
  filters << "scale=" << expectedOutputWidth << ":" << expectedOutputHeight;
  filters << ":sws_flags=bilinear";

  return std::make_unique<FiltersContext>(
      avFrame->width,
      avFrame->height,
      frameFormat,
      avFrame->sample_aspect_ratio,
      expectedOutputWidth,
      expectedOutputHeight,
      AV_PIX_FMT_RGB24,
      filters.str(),
      timeBase);
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
    const VideoStreamOptions& videoStreamOptions,
    UniqueAVFrame& avFrame,
    FrameOutput& frameOutput,
    std::optional<torch::Tensor> preAllocatedOutputTensor) {
  auto frameDims =
      getHeightAndWidthFromOptionsOrAVFrame(videoStreamOptions, avFrame);
  int expectedOutputHeight = frameDims.height;
  int expectedOutputWidth = frameDims.width;

  if (preAllocatedOutputTensor.has_value()) {
    auto shape = preAllocatedOutputTensor.value().sizes();
    TORCH_CHECK(
        (shape.size() == 3) && (shape[0] == expectedOutputHeight) &&
            (shape[1] == expectedOutputWidth) && (shape[2] == 3),
        "Expected pre-allocated tensor of shape ",
        expectedOutputHeight,
        "x",
        expectedOutputWidth,
        "x3, got ",
        shape);
  }

  torch::Tensor outputTensor;
  enum AVPixelFormat frameFormat =
      static_cast<enum AVPixelFormat>(avFrame->format);

  ColorConversionLibrary colorConversionLibrary =
      getColorConversionLibrary(videoStreamOptions, expectedOutputWidth);

  if (colorConversionLibrary == ColorConversionLibrary::SWSCALE) {
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
        expectedOutputWidth,
        expectedOutputHeight);

    outputTensor = preAllocatedOutputTensor.value_or(allocateEmptyHWCTensor(
        expectedOutputHeight, expectedOutputWidth, torch::kCPU));

    if (!swsContext_ || prevSwsFrameContext_ != swsFrameContext) {
      createSwsContext(swsFrameContext, avFrame->colorspace);
      prevSwsFrameContext_ = swsFrameContext;
    }
    int resultHeight =
        convertAVFrameToTensorUsingSwsScale(avFrame, outputTensor);
    // If this check failed, it would mean that the frame wasn't reshaped to
    // the expected height.
    // TODO: Can we do the same check for width?
    TORCH_CHECK(
        resultHeight == expectedOutputHeight,
        "resultHeight != expectedOutputHeight: ",
        resultHeight,
        " != ",
        expectedOutputHeight);

    frameOutput.data = outputTensor;
  } else if (colorConversionLibrary == ColorConversionLibrary::FILTERGRAPH) {
    TORCH_CHECK_EQ(avFrame->format, AV_PIX_FMT_RGB24);

    std::vector<int64_t> shape = {expectedOutputHeight, expectedOutputWidth, 3};
    std::vector<int64_t> strides = {avFrame->linesize[0], 3, 1};
    AVFrame* avFramePtr = avFrame.release();
    auto deleter = [avFramePtr](void*) {
      UniqueAVFrame avFrameToDelete(avFramePtr);
    };
    outputTensor = torch::from_blob(
        avFramePtr->data[0], shape, strides, deleter, {torch::kUInt8});

    if (preAllocatedOutputTensor.has_value()) {
      // We have already validated that preAllocatedOutputTensor and
      // outputTensor have the same shape.
      preAllocatedOutputTensor.value().copy_(outputTensor);
      frameOutput.data = preAllocatedOutputTensor.value();
    } else {
      frameOutput.data = outputTensor;
    }
  }
}

int CpuDeviceInterface::convertAVFrameToTensorUsingSwsScale(
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
