// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/torchcodec/_core/CpuDeviceInterface.h"

extern "C" {
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
}

namespace facebook::torchcodec {
namespace {

bool g_cpu = registerDeviceInterface(
    torch::kCPU,
    [](const torch::Device& device) { return new CpuDeviceInterface(device); });

} // namespace

bool CpuDeviceInterface::DecodedFrameContext::operator==(
    const CpuDeviceInterface::DecodedFrameContext& other) {
  return decodedWidth == other.decodedWidth &&
      decodedHeight == other.decodedHeight &&
      decodedFormat == other.decodedFormat &&
      expectedWidth == other.expectedWidth &&
      expectedHeight == other.expectedHeight;
}

bool CpuDeviceInterface::DecodedFrameContext::operator!=(
    const CpuDeviceInterface::DecodedFrameContext& other) {
  return !(*this == other);
}

CpuDeviceInterface::CpuDeviceInterface(const torch::Device& device)
    : DeviceInterface(device) {
  if (device_.type() != torch::kCPU) {
    throw std::runtime_error("Unsupported device: " + device_.str());
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
    const VideoStreamOptions& videoStreamOptions,
    const AVRational& timeBase,
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
  // We need to compare the current frame context with our previous frame
  // context. If they are different, then we need to re-create our colorspace
  // conversion objects. We create our colorspace conversion objects late so
  // that we don't have to depend on the unreliable metadata in the header.
  // And we sometimes re-create them because it's possible for frame
  // resolution to change mid-stream. Finally, we want to reuse the colorspace
  // conversion objects as much as possible for performance reasons.
  enum AVPixelFormat frameFormat =
      static_cast<enum AVPixelFormat>(avFrame->format);
  auto frameContext = DecodedFrameContext{
      avFrame->width,
      avFrame->height,
      frameFormat,
      avFrame->sample_aspect_ratio,
      expectedOutputWidth,
      expectedOutputHeight};

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
  auto defaultLibrary = (expectedOutputWidth % 32 == 0)
      ? ColorConversionLibrary::SWSCALE
      : ColorConversionLibrary::FILTERGRAPH;

  ColorConversionLibrary colorConversionLibrary =
      videoStreamOptions.colorConversionLibrary.value_or(defaultLibrary);

  if (colorConversionLibrary == ColorConversionLibrary::SWSCALE) {
    outputTensor = preAllocatedOutputTensor.value_or(allocateEmptyHWCTensor(
        expectedOutputHeight, expectedOutputWidth, torch::kCPU));

    if (!swsContext_ || prevFrameContext_ != frameContext) {
      createSwsContext(frameContext, avFrame->colorspace);
      prevFrameContext_ = frameContext;
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
    if (!filterGraphContext_.filterGraph || prevFrameContext_ != frameContext) {
      createFilterGraph(frameContext, videoStreamOptions, timeBase);
      prevFrameContext_ = frameContext;
    }
    outputTensor = convertAVFrameToTensorUsingFilterGraph(avFrame);

    // Similarly to above, if this check fails it means the frame wasn't
    // reshaped to its expected dimensions by filtergraph.
    auto shape = outputTensor.sizes();
    TORCH_CHECK(
        (shape.size() == 3) && (shape[0] == expectedOutputHeight) &&
            (shape[1] == expectedOutputWidth) && (shape[2] == 3),
        "Expected output tensor of shape ",
        expectedOutputHeight,
        "x",
        expectedOutputWidth,
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
    throw std::runtime_error(
        "Invalid color conversion library: " +
        std::to_string(static_cast<int>(colorConversionLibrary)));
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

torch::Tensor CpuDeviceInterface::convertAVFrameToTensorUsingFilterGraph(
    const UniqueAVFrame& avFrame) {
  int status = av_buffersrc_write_frame(
      filterGraphContext_.sourceContext, avFrame.get());
  if (status < AVSUCCESS) {
    throw std::runtime_error("Failed to add frame to buffer source context");
  }

  UniqueAVFrame filteredAVFrame(av_frame_alloc());
  status = av_buffersink_get_frame(
      filterGraphContext_.sinkContext, filteredAVFrame.get());
  TORCH_CHECK_EQ(filteredAVFrame->format, AV_PIX_FMT_RGB24);

  auto frameDims = getHeightAndWidthFromResizedAVFrame(*filteredAVFrame.get());
  int height = frameDims.height;
  int width = frameDims.width;
  std::vector<int64_t> shape = {height, width, 3};
  std::vector<int64_t> strides = {filteredAVFrame->linesize[0], 3, 1};
  AVFrame* filteredAVFramePtr = filteredAVFrame.release();
  auto deleter = [filteredAVFramePtr](void*) {
    UniqueAVFrame avFrameToDelete(filteredAVFramePtr);
  };
  return torch::from_blob(
      filteredAVFramePtr->data[0], shape, strides, deleter, {torch::kUInt8});
}

void CpuDeviceInterface::createFilterGraph(
    const DecodedFrameContext& frameContext,
    const VideoStreamOptions& videoStreamOptions,
    const AVRational& timeBase) {
  filterGraphContext_.filterGraph.reset(avfilter_graph_alloc());
  TORCH_CHECK(filterGraphContext_.filterGraph.get() != nullptr);

  if (videoStreamOptions.ffmpegThreadCount.has_value()) {
    filterGraphContext_.filterGraph->nb_threads =
        videoStreamOptions.ffmpegThreadCount.value();
  }

  const AVFilter* buffersrc = avfilter_get_by_name("buffer");
  const AVFilter* buffersink = avfilter_get_by_name("buffersink");

  std::stringstream filterArgs;
  filterArgs << "video_size=" << frameContext.decodedWidth << "x"
             << frameContext.decodedHeight;
  filterArgs << ":pix_fmt=" << frameContext.decodedFormat;
  filterArgs << ":time_base=" << timeBase.num << "/" << timeBase.den;
  filterArgs << ":pixel_aspect=" << frameContext.decodedAspectRatio.num << "/"
             << frameContext.decodedAspectRatio.den;

  int status = avfilter_graph_create_filter(
      &filterGraphContext_.sourceContext,
      buffersrc,
      "in",
      filterArgs.str().c_str(),
      nullptr,
      filterGraphContext_.filterGraph.get());
  if (status < 0) {
    throw std::runtime_error(
        std::string("Failed to create filter graph: ") + filterArgs.str() +
        ": " + getFFMPEGErrorStringFromErrorCode(status));
  }

  status = avfilter_graph_create_filter(
      &filterGraphContext_.sinkContext,
      buffersink,
      "out",
      nullptr,
      nullptr,
      filterGraphContext_.filterGraph.get());
  if (status < 0) {
    throw std::runtime_error(
        "Failed to create filter graph: " +
        getFFMPEGErrorStringFromErrorCode(status));
  }

  enum AVPixelFormat pix_fmts[] = {AV_PIX_FMT_RGB24, AV_PIX_FMT_NONE};

  status = av_opt_set_int_list(
      filterGraphContext_.sinkContext,
      "pix_fmts",
      pix_fmts,
      AV_PIX_FMT_NONE,
      AV_OPT_SEARCH_CHILDREN);
  if (status < 0) {
    throw std::runtime_error(
        "Failed to set output pixel formats: " +
        getFFMPEGErrorStringFromErrorCode(status));
  }

  UniqueAVFilterInOut outputs(avfilter_inout_alloc());
  UniqueAVFilterInOut inputs(avfilter_inout_alloc());

  outputs->name = av_strdup("in");
  outputs->filter_ctx = filterGraphContext_.sourceContext;
  outputs->pad_idx = 0;
  outputs->next = nullptr;
  inputs->name = av_strdup("out");
  inputs->filter_ctx = filterGraphContext_.sinkContext;
  inputs->pad_idx = 0;
  inputs->next = nullptr;

  std::stringstream description;
  description << "scale=" << frameContext.expectedWidth << ":"
              << frameContext.expectedHeight;
  description << ":sws_flags=bilinear";

  AVFilterInOut* outputsTmp = outputs.release();
  AVFilterInOut* inputsTmp = inputs.release();
  status = avfilter_graph_parse_ptr(
      filterGraphContext_.filterGraph.get(),
      description.str().c_str(),
      &inputsTmp,
      &outputsTmp,
      nullptr);
  outputs.reset(outputsTmp);
  inputs.reset(inputsTmp);
  if (status < 0) {
    throw std::runtime_error(
        "Failed to parse filter description: " +
        getFFMPEGErrorStringFromErrorCode(status));
  }

  status =
      avfilter_graph_config(filterGraphContext_.filterGraph.get(), nullptr);
  if (status < 0) {
    throw std::runtime_error(
        "Failed to configure filter graph: " +
        getFFMPEGErrorStringFromErrorCode(status));
  }
}

void CpuDeviceInterface::createSwsContext(
    const DecodedFrameContext& frameContext,
    const enum AVColorSpace colorspace) {
  SwsContext* swsContext = sws_getContext(
      frameContext.decodedWidth,
      frameContext.decodedHeight,
      frameContext.decodedFormat,
      frameContext.expectedWidth,
      frameContext.expectedHeight,
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
