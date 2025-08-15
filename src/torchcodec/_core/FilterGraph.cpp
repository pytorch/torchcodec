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

bool DecodedFrameContext::operator==(const DecodedFrameContext& other) {
  return decodedWidth == other.decodedWidth &&
      decodedHeight == other.decodedHeight &&
      decodedFormat == other.decodedFormat &&
      expectedWidth == other.expectedWidth &&
      expectedHeight == other.expectedHeight;
}

bool DecodedFrameContext::operator!=(const DecodedFrameContext& other) {
  return !(*this == other);
}

FilterGraph::FilterGraph(
    const DecodedFrameContext& frameContext,
    const VideoStreamOptions& videoStreamOptions,
    const AVRational& timeBase) {
  filterGraph_.reset(avfilter_graph_alloc());
  TORCH_CHECK(filterGraph_.get() != nullptr);

  if (videoStreamOptions.ffmpegThreadCount.has_value()) {
    filterGraph_->nb_threads = videoStreamOptions.ffmpegThreadCount.value();
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
      &sourceContext_,
      buffersrc,
      "in",
      filterArgs.str().c_str(),
      nullptr,
      filterGraph_.get());
  TORCH_CHECK(
      status >= 0,
      "Failed to create filter graph: ",
      filterArgs.str(),
      ": ",
      getFFMPEGErrorStringFromErrorCode(status));

  status = avfilter_graph_create_filter(
      &sinkContext_, buffersink, "out", nullptr, nullptr, filterGraph_.get());
  TORCH_CHECK(
      status >= 0,
      "Failed to create filter graph: ",
      getFFMPEGErrorStringFromErrorCode(status));

  enum AVPixelFormat pix_fmts[] = {AV_PIX_FMT_RGB24, AV_PIX_FMT_NONE};

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

  std::stringstream description;
  description << "scale=" << frameContext.expectedWidth << ":"
              << frameContext.expectedHeight;
  description << ":sws_flags=bilinear";

  AVFilterInOut* outputsTmp = outputs.release();
  AVFilterInOut* inputsTmp = inputs.release();
  status = avfilter_graph_parse_ptr(
      filterGraph_.get(),
      description.str().c_str(),
      &inputsTmp,
      &outputsTmp,
      nullptr);
  outputs.reset(outputsTmp);
  inputs.reset(inputsTmp);
  TORCH_CHECK(
      status >= 0,
      "Failed to parse filter description: ",
      getFFMPEGErrorStringFromErrorCode(status));

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
      status >= AVSUCCESS, "Failed to fet frame from buffer sink context");
  TORCH_CHECK_EQ(filteredAVFrame->format, AV_PIX_FMT_RGB24);

  return filteredAVFrame;
}

} // namespace facebook::torchcodec
