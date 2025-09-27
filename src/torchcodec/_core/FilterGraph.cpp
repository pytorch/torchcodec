// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/torchcodec/_core/FilterGraph.h"

extern "C" {
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
}

namespace facebook::torchcodec {

FiltersContext::FiltersContext(
    int inputWidth,
    int inputHeight,
    AVPixelFormat inputFormat,
    AVRational inputAspectRatio,
    int outputWidth,
    int outputHeight,
    AVPixelFormat outputFormat,
    const std::string& filtergraphStr,
    AVRational timeBase,
    AVBufferRef* hwFramesCtx)
    : inputWidth(inputWidth),
      inputHeight(inputHeight),
      inputFormat(inputFormat),
      inputAspectRatio(inputAspectRatio),
      outputWidth(outputWidth),
      outputHeight(outputHeight),
      outputFormat(outputFormat),
      filtergraphStr(filtergraphStr),
      timeBase(timeBase),
      hwFramesCtx(hwFramesCtx) {}

bool operator==(const AVRational& lhs, const AVRational& rhs) {
  return lhs.num == rhs.num && lhs.den == rhs.den;
}

bool FiltersContext::operator==(const FiltersContext& other) const {
  return inputWidth == other.inputWidth && inputHeight == other.inputHeight &&
      inputFormat == other.inputFormat && outputWidth == other.outputWidth &&
      outputHeight == other.outputHeight &&
      outputFormat == other.outputFormat &&
      filtergraphStr == other.filtergraphStr && timeBase == other.timeBase &&
      hwFramesCtx.get() == other.hwFramesCtx.get();
}

bool FiltersContext::operator!=(const FiltersContext& other) const {
  return !(*this == other);
}

FilterGraph::FilterGraph(
    const FiltersContext& filtersContext,
    const VideoStreamOptions& videoStreamOptions) {
  filterGraph_.reset(avfilter_graph_alloc());
  TORCH_CHECK(filterGraph_.get() != nullptr);

  if (videoStreamOptions.ffmpegThreadCount.has_value()) {
    filterGraph_->nb_threads = videoStreamOptions.ffmpegThreadCount.value();
  }

  const AVFilter* buffersrc = avfilter_get_by_name("buffer");
  const AVFilter* buffersink = avfilter_get_by_name("buffersink");

  UniqueAVBufferSrcParameters srcParams(av_buffersrc_parameters_alloc());
  TORCH_CHECK(srcParams, "Failed to allocate buffersrc params");

  srcParams->format = filtersContext.inputFormat;
  srcParams->width = filtersContext.inputWidth;
  srcParams->height = filtersContext.inputHeight;
  srcParams->sample_aspect_ratio = filtersContext.inputAspectRatio;
  srcParams->time_base = filtersContext.timeBase;
  if (filtersContext.hwFramesCtx) {
    srcParams->hw_frames_ctx = av_buffer_ref(filtersContext.hwFramesCtx.get());
  }

  sourceContext_ =
      avfilter_graph_alloc_filter(filterGraph_.get(), buffersrc, "in");
  TORCH_CHECK(sourceContext_, "Failed to allocate filter graph");

  int status = av_buffersrc_parameters_set(sourceContext_, srcParams.get());
  TORCH_CHECK(
      status >= 0,
      "Failed to create filter graph: ",
      getFFMPEGErrorStringFromErrorCode(status));

  status = avfilter_init_str(sourceContext_, nullptr);
  TORCH_CHECK(
      status >= 0,
      "Failed to create filter graph : ",
      getFFMPEGErrorStringFromErrorCode(status));

  status = avfilter_graph_create_filter(
      &sinkContext_, buffersink, "out", nullptr, nullptr, filterGraph_.get());
  TORCH_CHECK(
      status >= 0,
      "Failed to create filter graph: ",
      getFFMPEGErrorStringFromErrorCode(status));

  enum AVPixelFormat pix_fmts[] = {
      filtersContext.outputFormat, AV_PIX_FMT_NONE};

  status = av_opt_set_int_list(
      sinkContext_,
      "pix_fmts",
      pix_fmts,
      AV_PIX_FMT_NONE,
      AV_OPT_SEARCH_CHILDREN);
  TORCH_CHECK(
      status >= 0,
      "Failed to set output pixel formats: ",
      getFFMPEGErrorStringFromErrorCode(status));

  UniqueAVFilterInOut outputs(avfilter_inout_alloc());
  UniqueAVFilterInOut inputs(avfilter_inout_alloc());

  outputs->name = av_strdup("in");
  outputs->filter_ctx = sourceContext_;
  outputs->pad_idx = 0;
  outputs->next = nullptr;
  inputs->name = av_strdup("out");
  inputs->filter_ctx = sinkContext_;
  inputs->pad_idx = 0;
  inputs->next = nullptr;

  AVFilterInOut* outputsTmp = outputs.release();
  AVFilterInOut* inputsTmp = inputs.release();
  status = avfilter_graph_parse_ptr(
      filterGraph_.get(),
      filtersContext.filtergraphStr.c_str(),
      &inputsTmp,
      &outputsTmp,
      nullptr);
  outputs.reset(outputsTmp);
  inputs.reset(inputsTmp);
  TORCH_CHECK(
      status >= 0,
      "Failed to parse filter description: ",
      getFFMPEGErrorStringFromErrorCode(status),
      ", provided filters: " + filtersContext.filtergraphStr);

  status = avfilter_graph_config(filterGraph_.get(), nullptr);
  TORCH_CHECK(
      status >= 0,
      "Failed to configure filter graph: ",
      getFFMPEGErrorStringFromErrorCode(status));
}

UniqueAVFrame FilterGraph::convert(const UniqueAVFrame& avFrame) {
  int status = av_buffersrc_write_frame(sourceContext_, avFrame.get());
  TORCH_CHECK(
      status >= AVSUCCESS, "Failed to add frame to buffer source context");

  UniqueAVFrame filteredAVFrame(av_frame_alloc());
  status = av_buffersink_get_frame(sinkContext_, filteredAVFrame.get());
  TORCH_CHECK(
      status >= AVSUCCESS, "Failed to get frame from buffer sink context");

  return filteredAVFrame;
}

} // namespace facebook::torchcodec
