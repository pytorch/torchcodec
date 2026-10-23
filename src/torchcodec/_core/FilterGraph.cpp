// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "FilterGraph.h"
#include "FFMPEGCommon.h"
#include "StableABICompat.h"

extern "C" {
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
}

namespace facebook::torchcodec {

FiltersConfig::FiltersConfig(
    int input_width,
    int input_height,
    AVPixelFormat input_format,
    AVRational input_aspect_ratio,
    int output_width,
    int output_height,
    AVPixelFormat output_format,
    const std::string& filtergraph_str,
    AVRational time_base,
    AVBufferRef* hw_frames_ctx)
    : input_width(input_width),
      input_height(input_height),
      input_format(input_format),
      input_aspect_ratio(input_aspect_ratio),
      output_width(output_width),
      output_height(output_height),
      output_format(output_format),
      filtergraph_str(filtergraph_str),
      time_base(time_base),
      hw_frames_ctx(hw_frames_ctx) {}

bool operator==(const AVRational& lhs, const AVRational& rhs) {
  return lhs.num == rhs.num && lhs.den == rhs.den;
}

bool FiltersConfig::operator==(const FiltersConfig& other) const {
  return input_width == other.input_width &&
      input_height == other.input_height &&
      input_format == other.input_format &&
      output_width == other.output_width &&
      output_height == other.output_height &&
      output_format == other.output_format &&
      filtergraph_str == other.filtergraph_str &&
      time_base == other.time_base &&
      hw_frames_ctx.get() == other.hw_frames_ctx.get();
}

bool FiltersConfig::operator!=(const FiltersConfig& other) const {
  return !(*this == other);
}

FilterGraph::FilterGraph(
    const FiltersConfig& filters_config,
    const VideoStreamOptions& video_stream_options) {
  filter_graph_.reset(avfilter_graph_alloc());
  STD_TORCH_CHECK(
      filter_graph_.get() != nullptr, "Failed to allocate filter graph");

  if (video_stream_options.ffmpeg_thread_count.has_value()) {
    filter_graph_->nb_threads =
        video_stream_options.ffmpeg_thread_count.value();
  }

  // Configure the source context.
  const AVFilter* buffer_src = avfilter_get_by_name("buffer");
  UniqueAVBufferSrcParameters src_params(av_buffersrc_parameters_alloc());
  STD_TORCH_CHECK(src_params, "Failed to allocate buffersrc params");

  src_params->format = filters_config.input_format;
  src_params->width = filters_config.input_width;
  src_params->height = filters_config.input_height;
  src_params->sample_aspect_ratio = filters_config.input_aspect_ratio;
  src_params->time_base = filters_config.time_base;
  if (filters_config.hw_frames_ctx) {
    src_params->hw_frames_ctx =
        av_buffer_ref(filters_config.hw_frames_ctx.get());
  }

  source_context_ =
      avfilter_graph_alloc_filter(filter_graph_.get(), buffer_src, "in");
  STD_TORCH_CHECK(source_context_, "Failed to allocate filter graph");

  int status = av_buffersrc_parameters_set(source_context_, src_params.get());
  STD_TORCH_CHECK(
      status >= 0,
      "Failed to create filter graph: ",
      get_ffmpeg_error_string_from_error_code(status));

  status = avfilter_init_str(source_context_, nullptr);
  STD_TORCH_CHECK(
      status >= 0,
      "Failed to create filter graph : ",
      get_ffmpeg_error_string_from_error_code(status));

  // Configure the sink context.
  const AVFilter* buffer_sink = avfilter_get_by_name("buffersink");
  STD_TORCH_CHECK(buffer_sink != nullptr, "Failed to get buffersink filter.");

  sink_context_ = create_av_filter_context_with_options(
      filter_graph_.get(), buffer_sink, filters_config.output_format);
  STD_TORCH_CHECK(
      sink_context_ != nullptr, "Failed to create and configure buffersink");

  // Create the filtergraph nodes based on the source and sink contexts.
  UniqueAVFilterInOut outputs(avfilter_inout_alloc());
  outputs->name = av_strdup("in");
  outputs->filter_ctx = source_context_;
  outputs->pad_idx = 0;
  outputs->next = nullptr;

  UniqueAVFilterInOut inputs(avfilter_inout_alloc());
  inputs->name = av_strdup("out");
  inputs->filter_ctx = sink_context_;
  inputs->pad_idx = 0;
  inputs->next = nullptr;

  // Create the filtergraph specified by the filtergraph string in the context
  // of the inputs and outputs. Note the dance we have to do with release and
  // resetting the output and input nodes because FFmpeg modifies them in place.
  AVFilterInOut* outputs_tmp = outputs.release();
  AVFilterInOut* inputs_tmp = inputs.release();
  status = avfilter_graph_parse_ptr(
      filter_graph_.get(),
      filters_config.filtergraph_str.c_str(),
      &inputs_tmp,
      &outputs_tmp,
      nullptr);
  outputs.reset(outputs_tmp);
  inputs.reset(inputs_tmp);
  STD_TORCH_CHECK(
      status >= 0,
      "Failed to parse filter description: ",
      get_ffmpeg_error_string_from_error_code(status),
      ", provided filters: " + filters_config.filtergraph_str);

  // Check filtergraph validity and configure links and formats.
  status = avfilter_graph_config(filter_graph_.get(), nullptr);
  STD_TORCH_CHECK(
      status >= 0,
      "Failed to configure filter graph: ",
      get_ffmpeg_error_string_from_error_code(status),
      ", provided filters: " + filters_config.filtergraph_str);
}

namespace {

bool is_reserved(AVColorPrimaries primaries) {
  return primaries == AVCOL_PRI_RESERVED0 || primaries == AVCOL_PRI_RESERVED ||
      av_color_primaries_name(primaries) == nullptr;
}

bool is_reserved(AVColorTransferCharacteristic trc) {
  return trc == AVCOL_TRC_RESERVED0 || trc == AVCOL_TRC_RESERVED ||
      av_color_transfer_name(trc) == nullptr;
}

bool is_reserved(AVColorSpace colorspace) {
  return colorspace == AVCOL_SPC_RESERVED ||
      av_color_space_name(colorspace) == nullptr;
}

} // namespace

UniqueAVFrame FilterGraph::convert(const AVFrame& av_frame) {
  // From FFmpeg >= 8, swscale (used by the scale filter) rejects frames with
  // reserved color properties. FFmpeg < 8 ignores them. A 'reserved' value is
  // argulably a bad encoding. We just preserve the FFmpeg < 8 behavior by
  // setting them to unspecified, which ignores them.
  const AVFrame* input_frame = &av_frame;
  UniqueAVFrame sanitized_frame;
  if (is_reserved(av_frame.color_primaries) ||
      is_reserved(av_frame.color_trc) || is_reserved(av_frame.colorspace)) {
    sanitized_frame.reset(av_frame_clone(&av_frame));
    STD_TORCH_CHECK(sanitized_frame != nullptr, "Failed to clone frame");
    if (is_reserved(sanitized_frame->color_primaries)) {
      sanitized_frame->color_primaries = AVCOL_PRI_UNSPECIFIED;
    }
    if (is_reserved(sanitized_frame->color_trc)) {
      sanitized_frame->color_trc = AVCOL_TRC_UNSPECIFIED;
    }
    if (is_reserved(sanitized_frame->colorspace)) {
      sanitized_frame->colorspace = AVCOL_SPC_UNSPECIFIED;
    }
    input_frame = sanitized_frame.get();
  }

  int status = av_buffersrc_write_frame(source_context_, input_frame);
  STD_TORCH_CHECK(
      status >= AVSUCCESS,
      "Failed to add frame to buffer source context: ",
      get_ffmpeg_error_string_from_error_code(status));

  UniqueAVFrame filtered_av_frame(av_frame_alloc());
  status = av_buffersink_get_frame(sink_context_, filtered_av_frame.get());
  STD_TORCH_CHECK(
      status >= AVSUCCESS,
      "Failed to get frame from buffer sink context: ",
      get_ffmpeg_error_string_from_error_code(status));

  return filtered_av_frame;
}

} // namespace facebook::torchcodec
