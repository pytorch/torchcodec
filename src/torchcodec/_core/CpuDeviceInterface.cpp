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
    [[maybe_unused]] const FrameDims& metadataDims,
    const std::optional<FrameDims>& resizedOutputDims) {
  videoStreamOptions_ = videoStreamOptions;
  timeBase_ = timeBase;
  resizedOutputDims_ = resizedOutputDims;

  // We can only use swscale when we have a single resize transform. Note that
  // this means swscale will not support the case of having several,
  // back-to-base resizes. There's no strong reason to even do that, but if
  // someone does, it's more correct to implement that with filtergraph.
  //
  // We calculate this value during initilization but we don't refer to it until
  // getColorConversionLibrary() is called. Calculating this value during
  // initialization saves us from having to save all of the transforms.
  areTransformsSwScaleCompatible_ = transforms.empty() ||
      (transforms.size() == 1 && transforms[0]->isResize());

  // Note that we do not expose this capability in the public API, only through
  // the core API.
  //
  // Same as above, we calculate this value during initialization and refer to
  // it in getColorConversionLibrary().
  userRequestedSwScale_ = videoStreamOptions_.colorConversionLibrary ==
      ColorConversionLibrary::SWSCALE;

  // We can only use swscale when we have a single resize transform. Note that
  // we actually decide on whether or not to actually use swscale at the last
  // possible moment, when we actually convert the frame. This is because we
  // need to know the actual frame dimensions.
  if (transforms.size() == 1 && transforms[0]->isResize()) {
    auto resize = dynamic_cast<ResizeTransform*>(transforms[0].get());
    TORCH_CHECK(resize != nullptr, "ResizeTransform expected but not found!")
    swsFlags_ = resize->getSwsFlags();
  }

  // If we have any transforms, replace filters_ with the filter strings from
  // the transforms. As noted above, we decide between swscale and filtergraph
  // when we actually decode a frame.
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

  initialized_ = true;
}

ColorConversionLibrary CpuDeviceInterface::getColorConversionLibrary(
    const FrameDims& outputDims) {
  // swscale requires widths to be multiples of 32:
  // https://stackoverflow.com/questions/74351955/turn-off-sw-scale-conversion-to-planar-yuv-32-byte-alignment-requirements
  bool isWidthSwScaleCompatible = (outputDims.width % 32) == 0;

  // We want to use swscale for color conversion if possible because it is
  // faster than filtergraph. The following are the conditions we need to meet
  // to use it.
  //
  // Note that we treat the transform limitation differently from the width
  // limitation. That is, we consider the transforms being compatible with
  // swscale as a hard requirement. If the transforms are not compatiable,
  // then we will end up not applying the transforms, and that is wrong.
  //
  // The width requirement, however, is a soft requirement. Even if we don't
  // meet it, we let the user override it. We have tests that depend on this
  // behavior. Since we don't expose the ability to choose swscale or
  // filtergraph in our public API, this is probably okay. It's also the only
  // way that we can be certain we are testing one versus the other.
  if (areTransformsSwScaleCompatible_ &&
      (userRequestedSwScale_ || isWidthSwScaleCompatible)) {
    return ColorConversionLibrary::SWSCALE;
  } else {
    return ColorConversionLibrary::FILTERGRAPH;
  }
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

  // Note that we ignore the dimensions from the metadata; we don't even bother
  // storing them. The resized dimensions take priority. If we don't have any,
  // then we use the dimensions from the actual decoded frame. We use the actual
  // decoded frame and not the metadata for two reasons:
  //
  //   1. Metadata may be wrong. If we access to more accurate information, we
  //      should use it.
  //   2. Video streams can have variable resolution. This fact is not captured
  //      in the stream  metadata.
  //
  // Both cases cause problems for our batch APIs, as we allocate
  // FrameBatchOutputs based on the the stream metadata. But single-frame APIs
  // can still work in such situations, so they should.
  auto outputDims =
      resizedOutputDims_.value_or(FrameDims(avFrame->width, avFrame->height));

  if (preAllocatedOutputTensor.has_value()) {
    auto shape = preAllocatedOutputTensor.value().sizes();
    TORCH_CHECK(
        (shape.size() == 3) && (shape[0] == outputDims.height) &&
            (shape[1] == outputDims.width) && (shape[2] == 3),
        "Expected pre-allocated tensor of shape ",
        outputDims.height,
        "x",
        outputDims.width,
        "x3, got ",
        shape);
  }

  auto colorConversionLibrary = getColorConversionLibrary(outputDims);
  torch::Tensor outputTensor;
  enum AVPixelFormat frameFormat =
      static_cast<enum AVPixelFormat>(avFrame->format);

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
        outputDims.width,
        outputDims.height);

    outputTensor = preAllocatedOutputTensor.value_or(
        allocateEmptyHWCTensor(outputDims, torch::kCPU));

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
        resultHeight == outputDims.height,
        "resultHeight != outputDims.height: ",
        resultHeight,
        " != ",
        outputDims.height);

    frameOutput.data = outputTensor;
  } else if (colorConversionLibrary == ColorConversionLibrary::FILTERGRAPH) {
    FiltersContext filtersContext(
        avFrame->width,
        avFrame->height,
        frameFormat,
        avFrame->sample_aspect_ratio,
        outputDims.width,
        outputDims.height,
        AV_PIX_FMT_RGB24,
        filters_,
        timeBase_);

    if (!filterGraph_ || prevFiltersContext_ != filtersContext) {
      filterGraph_ =
          std::make_unique<FilterGraph>(filtersContext, videoStreamOptions_);
      prevFiltersContext_ = std::move(filtersContext);
    }
    outputTensor = rgbAVFrameToTensor(filterGraph_->convert(avFrame));

    // Similarly to above, if this check fails it means the frame wasn't
    // reshaped to its expected dimensions by filtergraph.
    auto shape = outputTensor.sizes();
    TORCH_CHECK(
        (shape.size() == 3) && (shape[0] == outputDims.height) &&
            (shape[1] == outputDims.width) && (shape[2] == 3),
        "Expected output tensor of shape ",
        outputDims.height,
        "x",
        outputDims.width,
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
        static_cast<int>(colorConversionLibrary));
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
