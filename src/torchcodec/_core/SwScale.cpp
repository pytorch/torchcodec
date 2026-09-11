// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "SwScale.h"
#include "Frame.h"

namespace facebook::torchcodec {

SwScale::SwScale(const SwsConfig& config, int sws_flags)
    : config_(config), sws_flags_(sws_flags) {
  needs_resize_ =
      (config_.input_height != config_.output_height ||
       config_.input_width != config_.output_width);

  // Create color conversion context (input format -> output RGB format).
  // Color conversion always outputs at the input resolution.
  // When no resize is needed, input and output resolutions are the same.
  // See [Transform and Format Conversion Order] for more on the output pixel
  // format.
  SwsConfig color_conversion_frame_config(
      config_.input_width,
      config_.input_height,
      config_.input_format,
      config_.input_colorspace,
      config_.input_width,
      config_.input_height,
      config_.output_format);
  color_conversion_frame_config.input_color_range = config_.input_color_range;

  color_conversion_sws_context_ = create_sws_context(
      color_conversion_frame_config,
      // No flags for color conversion. When resizing is needed, we use a
      // separate swscale context with the appropriate resize flags.
      /*swsFlags=*/0);

  // Create resize context if needed (output RGB at input resolution ->
  // output RGB at output resolution).
  if (needs_resize_) {
    SwsConfig resize_frame_config(
        config_.input_width,
        config_.input_height,
        config_.output_format,
        AVCOL_SPC_RGB,
        config_.output_width,
        config_.output_height,
        config_.output_format);

    resize_sws_context_ = create_sws_context(resize_frame_config, sws_flags_);
  }
}

int SwScale::convert(
    const AVFrame& av_frame,
    torch::stable::Tensor& output_tensor) {
  // When resizing is needed, we do sws_scale twice: first convert to output
  // RGB at original resolution, then resize in output RGB space. This ensures
  // transforms happen in the output color space (RGB) rather than the input
  // color space (YUV).
  //
  // When no resize is needed, we do color conversion directly into the output
  // tensor.
  // RGB24 = 3 channels x 1 byte (uint8); RGB48 = 3 channels x 2 bytes (uint16).
  bool is_rgb48 = config_.output_format == AV_PIX_FMT_RGB48;
  int bytes_per_pixel = is_rgb48 ? 6 : 3;
  auto output_dtype = is_rgb48 ? OutputDtype::FLOAT32 : OutputDtype::UINT8;
  torch::stable::Tensor color_converted_tensor = needs_resize_
      ? allocate_empty_hwc_tensor(
            FrameDims(config_.input_height, config_.input_width),
            kStableCPU,
            output_dtype)
      : output_tensor;

  // sws_scale always takes uint8_t* pointers regardless of actual bit depth.
  uint8_t* color_converted_pointers[4] = {
      static_cast<uint8_t*>(color_converted_tensor.mutable_data_ptr()),
      nullptr,
      nullptr,
      nullptr};
  int color_converted_width =
      static_cast<int>(color_converted_tensor.sizes()[1]);
  int color_converted_linesizes[4] = {
      color_converted_width * bytes_per_pixel, 0, 0, 0};

  int color_converted_height = sws_scale(
      color_conversion_sws_context_.get(),
      av_frame.data,
      av_frame.linesize,
      0,
      av_frame.height,
      color_converted_pointers,
      color_converted_linesizes);

  STD_TORCH_CHECK(
      color_converted_height == av_frame.height,
      "Color conversion swscale pass failed: colorConvertedHeight != avFrame->height: ",
      color_converted_height,
      " != ",
      av_frame.height);

  if (needs_resize_) {
    uint8_t* src_pointers[4] = {
        static_cast<uint8_t*>(color_converted_tensor.mutable_data_ptr()),
        nullptr,
        nullptr,
        nullptr};
    int src_linesizes[4] = {config_.input_width * bytes_per_pixel, 0, 0, 0};

    uint8_t* dst_pointers[4] = {
        static_cast<uint8_t*>(output_tensor.mutable_data_ptr()),
        nullptr,
        nullptr,
        nullptr};
    int expected_output_width = static_cast<int>(output_tensor.sizes()[1]);
    int dst_linesizes[4] = {expected_output_width * bytes_per_pixel, 0, 0, 0};

    color_converted_height = sws_scale(
        resize_sws_context_.get(),
        src_pointers,
        src_linesizes,
        0,
        config_.input_height,
        dst_pointers,
        dst_linesizes);
  }

  return color_converted_height;
}

} // namespace facebook::torchcodec
