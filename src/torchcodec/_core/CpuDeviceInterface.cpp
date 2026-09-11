// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "CpuDeviceInterface.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include "AudioCommon.h"

namespace facebook::torchcodec {

namespace {

AVPixelFormat validate_pixel_format(
    const AVCodec& av_codec,
    const std::string& target_pixel_format) {
  AVPixelFormat pixel_format = av_get_pix_fmt(target_pixel_format.c_str());
  const AVPixelFormat* supported_formats =
      get_supported_pixel_formats(av_codec);
  if (supported_formats != nullptr) {
    for (int i = 0; supported_formats[i] != AV_PIX_FMT_NONE; ++i) {
      if (supported_formats[i] == pixel_format) {
        return pixel_format;
      }
    }
  }
  std::stringstream error_msg;
  if (pixel_format == AV_PIX_FMT_NONE) {
    error_msg << "Unknown pixel format: " << target_pixel_format;
  } else {
    error_msg << "Specified pixel format " << target_pixel_format
              << " is not supported by the " << av_codec.name << " encoder.";
  }
  error_msg << "\nSupported pixel formats for " << av_codec.name << ":";
  if (supported_formats != nullptr) {
    for (int i = 0; supported_formats[i] != AV_PIX_FMT_NONE; ++i) {
      error_msg << " " << av_get_pix_fmt_name(supported_formats[i]);
    }
  }
  STD_TORCH_CHECK(false, error_msg.str());
}

AVPixelFormat get_output_pixel_format(OutputDtype output_dtype) {
  return output_dtype == OutputDtype::FLOAT32 ? AV_PIX_FMT_RGB48
                                              : AV_PIX_FMT_RGB24;
}

// Returns the format filter string for the given output pixel format.
std::string get_format_filter_string(AVPixelFormat output_format) {
  return (output_format == AV_PIX_FMT_RGB48) ? "format=rgb48,"
                                             : "format=rgb24,";
}

static bool g_cpu = register_device_interface(
    DeviceInterfaceKey(kStableCPU),
    [](const StableDevice& device) { return new CpuDeviceInterface(device); });

} // namespace

AVPixelFormat CpuDeviceInterface::get_encoding_pixel_format(
    const AVCodec& av_codec,
    const std::optional<std::string>& user_pixel_format) const {
  if (user_pixel_format.has_value()) {
    return validate_pixel_format(av_codec, user_pixel_format.value());
  }
  const AVPixelFormat* formats = get_supported_pixel_formats(av_codec);
  // Pick the codec's first supported format (often yuv420p), else fall back.
  return (formats && formats[0] != AV_PIX_FMT_NONE) ? formats[0]
                                                    : AV_PIX_FMT_YUV420P;
}

CpuDeviceInterface::CpuDeviceInterface(const StableDevice& device)
    : DeviceInterface(device) {
  STD_TORCH_CHECK(g_cpu, "CpuDeviceInterface was not registered!");
  STD_TORCH_CHECK(
      device_.type() == kStableCPU, "Unsupported device: must be CPU");
}

void CpuDeviceInterface::initialize(const SharedAVCodecContext& codec_context) {
  codec_context_ = codec_context;
}

void CpuDeviceInterface::initialize_video_decoding(
    const AVStream* av_stream,
    [[maybe_unused]] const UniqueDecodingAVFormatContext& av_format_ctx,
    [[maybe_unused]] const VideoStreamOptions& video_stream_options) {
  STD_TORCH_CHECK(av_stream != nullptr, "avStream is null");
  time_base_ = av_stream->time_base;
}

void CpuDeviceInterface::initialize_color_conversion(
    const VideoStreamOptions& video_stream_options,
    const std::vector<std::unique_ptr<Transform>>& transforms,
    const std::optional<FrameDims>& resized_output_dims) {
  av_media_type_ = AVMEDIA_TYPE_VIDEO;
  video_stream_options_ = video_stream_options;
  resized_output_dims_ = resized_output_dims;
  output_pixel_format_ =
      get_output_pixel_format(video_stream_options_.output_dtype);

  // We can use swscale when we have a single resize transform.
  // With a single resize, we use swscale twice:
  // first for color conversion (YUV->RGB24), then for resize in RGB24 space.
  //
  // Note that this means swscale will not support the case of having several,
  // back-to-back resizes or other transforms.
  //
  // We calculate this value during initialization but we don't refer to it
  // until getColorConversionLibrary() is called. Calculating this value during
  // initialization saves us from having to save all of the transforms.
  are_transforms_sw_scale_compatible_ = transforms.empty() ||
      (transforms.size() == 1 && transforms[0]->is_resize());

  // Note that we do not expose this capability in the public API, only through
  // the core API.
  //
  // Same as above, we calculate this value during initialization and refer to
  // it in getColorConversionLibrary().
  user_requested_sw_scale_ = video_stream_options_.color_conversion_library ==
      ColorConversionLibrary::SWSCALE;

  // We can only use swscale when we have a single resize transform. Note that
  // we actually decide on whether or not to actually use swscale at the last
  // possible moment, when we actually convert the frame. This is because we
  // need to know the actual frame dimensions.
  if (transforms.size() == 1 && transforms[0]->is_resize()) {
    auto resize = dynamic_cast<ResizeTransform*>(transforms[0].get());
    STD_TORCH_CHECK(
        resize != nullptr, "ResizeTransform expected but not found!");
    sws_flags_ = resize->get_sws_flags();
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
    filters << transform->get_filter_graph_cpu();
    first = false;
  }
  if (!transforms.empty()) {
    // Note [Transform and Format Conversion Order]
    // We have to ensure that all user filters happen AFTER the explicit format
    // conversion. That is, we want the filters to be applied in the output RGB
    // color space, not the pixel format of the input frame.
    //
    // The output frame will always be in the output RGB format (RGB24 or
    // RGB48), as we specify the sink node with the appropriate format.
    // Filtergraph will automatically insert a format conversion to ensure the
    // output frame matches the pixel format specified in the sink. But by
    // default, it will insert it after the user filters. We need an explicit
    // format conversion to get the behavior we want.
    filters_ = get_format_filter_string(output_pixel_format_) + filters.str();
  }

  initialized_ = true;
}

void CpuDeviceInterface::initialize_audio(
    const AudioStreamOptions& audio_stream_options,
    double stream_start_seconds) {
  av_media_type_ = AVMEDIA_TYPE_AUDIO;
  audio_stream_options_ = audio_stream_options;
  audio_stream_start_seconds_ = stream_start_seconds;
  initialized_ = true;
}

ColorConversionLibrary CpuDeviceInterface::get_color_conversion_library(
    const FrameDims& input_dims,
    const FrameDims& output_dims) const {
  // swscale requires widths to be multiples of 32:
  // https://stackoverflow.com/questions/74351955/turn-off-sw-scale-conversion-to-planar-yuv-32-byte-alignment-requirements
  bool are_widths_sw_scale_compatible =
      (input_dims.width % 32) == 0 && (output_dims.width % 32) == 0;

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
  if (are_transforms_sw_scale_compatible_ &&
      (user_requested_sw_scale_ || are_widths_sw_scale_compatible)) {
    return ColorConversionLibrary::SWSCALE;
  } else {
    return ColorConversionLibrary::FILTERGRAPH;
  }
}

void CpuDeviceInterface::convert_av_frame_to_frame_output(
    const AVFrame& av_frame,
    FrameOutput& frame_output,
    std::optional<torch::stable::Tensor> pre_allocated_output_tensor) {
  STD_TORCH_CHECK(initialized_, "CpuDeviceInterface was not initialized.");

  if (av_media_type_ == AVMEDIA_TYPE_AUDIO) {
    convert_audio_av_frame_to_frame_output(av_frame, frame_output);
  } else {
    convert_video_av_frame_to_frame_output(
        av_frame, frame_output, pre_allocated_output_tensor);
  }
}

// Note [preAllocatedOutputTensor with swscale and filtergraph]:
// Callers may pass a pre-allocated tensor, where the output.data tensor will
// be stored. This parameter is honored in any case, but it only leads to a
// speed-up when swscale is used. With swscale, we can tell ffmpeg to place the
// decoded frame directly into `preAllocatedtensor.mutable_data_ptr()`. We
// haven't yet found a way to do that with filtegraph.
// TODO: Figure out whether that's possible!
// Dimension order of the preAllocatedOutputTensor must be HWC, regardless of
// `dimension_order` parameter. It's up to callers to re-shape it if needed.
void CpuDeviceInterface::convert_video_av_frame_to_frame_output(
    const AVFrame& av_frame,
    FrameOutput& frame_output,
    std::optional<torch::stable::Tensor> pre_allocated_output_tensor) {
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
  auto input_dims = FrameDims(av_frame.height, av_frame.width);
  auto output_dims = resized_output_dims_.value_or(input_dims);

  if (pre_allocated_output_tensor.has_value()) {
    auto shape = pre_allocated_output_tensor.value().sizes();
    STD_TORCH_CHECK(
        (shape.size() == 3) && (shape[0] == output_dims.height) &&
            (shape[1] == output_dims.width) && (shape[2] == 3),
        "Expected pre-allocated tensor of shape ",
        output_dims.height,
        "x",
        output_dims.width,
        "x3, got ",
        int_array_ref_to_string(shape));
  }

  auto color_conversion_library =
      get_color_conversion_library(input_dims, output_dims);
  torch::stable::Tensor output_tensor;

  if (color_conversion_library == ColorConversionLibrary::SWSCALE) {
    output_tensor =
        pre_allocated_output_tensor.value_or(allocate_empty_hwc_tensor(
            output_dims, kStableCPU, video_stream_options_.output_dtype));

    auto av_frame_format = static_cast<AVPixelFormat>(av_frame.format);
    SwsConfig sws_config{
        .input_width = av_frame.width,
        .input_height = av_frame.height,
        .input_format = av_frame_format,
        .input_colorspace = av_frame.colorspace,
        .input_color_range = av_frame.color_range,
        .output_width = output_dims.width,
        .output_height = output_dims.height,
        .output_format = output_pixel_format_};

    if (!sw_scale_ || sw_scale_->get_config() != sws_config) {
      sw_scale_ = std::make_unique<SwScale>(sws_config, sws_flags_);
    }

    int result_height = sw_scale_->convert(av_frame, output_tensor);

    // If this check failed, it would mean that the frame wasn't reshaped to
    // the expected height.
    // TODO: Can we do the same check for width?
    STD_TORCH_CHECK(
        result_height == output_dims.height,
        "resultHeight != outputDims.height: ",
        result_height,
        " != ",
        output_dims.height);

    frame_output.data = output_tensor;
  } else if (color_conversion_library == ColorConversionLibrary::FILTERGRAPH) {
    output_tensor =
        convert_av_frame_to_tensor_using_filter_graph(av_frame, output_dims);

    // Similarly to above, if this check fails it means the frame wasn't
    // reshaped to its expected dimensions by filtergraph.
    auto shape = output_tensor.sizes();
    STD_TORCH_CHECK(
        (shape.size() == 3) && (shape[0] == output_dims.height) &&
            (shape[1] == output_dims.width) && (shape[2] == 3),
        "Expected output tensor of shape ",
        output_dims.height,
        "x",
        output_dims.width,
        "x3, got ",
        int_array_ref_to_string(shape));

    if (pre_allocated_output_tensor.has_value()) {
      // We have already validated that preAllocatedOutputTensor and
      // outputTensor have the same shape.
      torch::stable::copy_(pre_allocated_output_tensor.value(), output_tensor);
      frame_output.data = pre_allocated_output_tensor.value();
    } else {
      frame_output.data = output_tensor;
    }
  } else {
    STD_TORCH_CHECK(
        false,
        "Invalid color conversion library: ",
        static_cast<int>(color_conversion_library));
  }
}

torch::stable::Tensor
CpuDeviceInterface::convert_av_frame_to_tensor_using_filter_graph(
    const AVFrame& av_frame,
    const FrameDims& output_dims) {
  auto av_frame_format = static_cast<AVPixelFormat>(av_frame.format);

  FiltersConfig filters_config(
      av_frame.width,
      av_frame.height,
      av_frame_format,
      av_frame.sample_aspect_ratio,
      output_dims.width,
      output_dims.height,
      output_pixel_format_,
      filters_,
      time_base_);

  if (!filter_graph_ || prev_filters_config_ != filters_config) {
    filter_graph_ =
        std::make_unique<FilterGraph>(filters_config, video_stream_options_);
    prev_filters_config_ = std::move(filters_config);
  }
  return rgb_av_frame_to_tensor(*filter_graph_->convert(av_frame));
}

// clang-format off
// Note [Audio resampling and frame alignment]
//
// Say we have some audio at 24kHz that we resample at 16kHz. We want to honor
// the natural property that `get_all_samples()` should be equal to the
// concatenation of calls to `get_samples_played_in_range(a, b)` where all the
// (a, b) pairs are a full partition of the duration.
//
// `get_all_samples()` will return this:
//
// 0s                                                                                                                                                          1s
// 0                                                                             12                                                                            24
// ||   |   |   ||   |   |   ||   |   |   ||   |   |   ||   |   |   ||   |   |   ||   |   |   ||   |   |   ||   |   |   ||   |   |   ||   |   |   ||   |   |   ||     24Hz
// ||     |     ||     |     ||     |     ||     |     ||     |     ||     |     ||     |     ||     |     ||     |     ||     |     ||     |     ||     |     ||     16Hz
// 0                                                                              8                                                                            16
//
// Now, the calls to `get_samples_played_in_range(a, b)`. If we don't do
// something special, we'd get the following.
//
// 0s                x                                                                                                                                         1s
// 0                 [  sample frame  ]                                          12                                                                            24
// ||   |   |   ||   |   |   ||   |   |   ||   |   |   ||   |   |   ||   |   |   ||   |   |   ||   |   |   ||   |   |   ||   |   |   ||   |   |   ||   |   |   ||     24Hz
// ||     |     ||     |     ||     |     ||     |     ||     |     ||     |     ||     |     ||     |     ||     |     ||     |     ||     |     ||     |     ||     16Hz
// 0            ||   ^                                 ||                         8                                                                            16
//              ||   |
//              ||   x', not aligned with output grid!
//              ||
//              ||
//              ||
//              ||
//              we tell the resampler to start here! On a sample that is aligned in input and output space.
//
// (side note: the frame above is drawn as containing very few samples,
// real frames have much more, it doesn't matter for the point being made.)
//
// Say we request `get_samples_played_in_range(a, b)` where a is somewhere
// within the `[sample frame]` - it doesn't matter where it is exactly. Since
// FFmpeg starts decoding from a frame start, it will start decoding from x:
// the sample in the input space where that frame starts.
//
//
// **THE PROBLEM** is that this sample `x` in the input space is NOT aligned
// with a sample boundary in the output space! See how x' doesn't align with a
// sample that the single call to `get_all_samples()` would have returned?
// Nothing breaks, the decoder and the resampler happily oblige, but they start
// returning an output that is stamped with a pts that we would have otherwise
// never gotten - and all the resulting samples boundaries are shifted by that
// offset. This breaks our desired property.
//
// To fix this, we tell the resampler to NOT start at x, but to start instead
// at a sample that is aligned in both input space and output space: the ones
// that start with ||. We know where these samples are: their index is a
// multiple of k in input space, and of k' in output space, where:
//
// - k  = isr / gcd(isr, osr)   -- every 3 samples in input space, since
//                                 gcd = 8 and 24 = 8 * 3
// - k' = osr / gcd(isr, osr)   -- every 2 samples in output space, since
//                                 gcd = 8 and 16 = 8 * 2
//
// In practice, we tell the resampler to *drop* some samples. We can do that
// because in practice, we start feeding it much earlier than the target `a`,
// because of the pre-roll (see [Audio pre-roll and post-roll]). The
// pre-roll is large enough such that the aligned sample we land on is always at
// or before `a`. Whether it is before the frame containing `a`, as drawn above,
// or inside that frame depends on how far into that frame `a` falls - but it is
// never past `a`, so the samples we drop are never samples the caller asked
// for.
// clang-format on
void CpuDeviceInterface::convert_audio_av_frame_to_frame_output(
    const AVFrame& src_av_frame,
    FrameOutput& frame_output) {
  AVSampleFormat src_sample_format =
      static_cast<AVSampleFormat>(src_av_frame.format);
  AVSampleFormat out_sample_format = AV_SAMPLE_FMT_FLTP;

  int src_sample_rate = src_av_frame.sample_rate;
  int out_sample_rate =
      audio_stream_options_.sample_rate.value_or(src_sample_rate);

  int src_num_channels = get_num_channels(codec_context_);
  STD_TORCH_CHECK(
      src_num_channels == get_num_channels(src_av_frame),
      "The frame has ",
      get_num_channels(src_av_frame),
      " channels, expected ",
      src_num_channels,
      ". If you are hitting this, it may be because you are using "
      "a buggy FFmpeg version. FFmpeg4 is known to fail here in some "
      "valid scenarios. Try to upgrade FFmpeg?");
  int out_num_channels =
      audio_stream_options_.num_channels.value_or(src_num_channels);

  bool must_convert =
      (src_sample_format != out_sample_format ||
       src_sample_rate != out_sample_rate ||
       src_num_channels != out_num_channels);

  UniqueAVFrame converted_av_frame;
  if (must_convert) {
    int64_t num_samples_to_skip = 0;
    if (!swr_context_) {
      // See [Audio resampling and frame alignment].
      int64_t frame_sample_index = std::llround(
          (frame_output.pts_seconds - audio_stream_start_seconds_) *
          src_sample_rate);
      int64_t alignment = // k in the note
          src_sample_rate / std::gcd(src_sample_rate, out_sample_rate);
      int64_t remainder = frame_sample_index % alignment;
      if (remainder < 0) {
        remainder += alignment;
      }
      int64_t next_aligned_sample =
          frame_sample_index + (remainder == 0 ? 0 : alignment - remainder);
      num_samples_to_skip = next_aligned_sample - frame_sample_index;

      // The pts_seconds field was set upstream to the pts of src_av_frame, but
      // must adjust it to report the pts of the aligned sample, since this is
      // where the resampler will start producing output.
      frame_output.pts_seconds = audio_stream_start_seconds_ +
          static_cast<double>(next_aligned_sample) / src_sample_rate;

      if (num_samples_to_skip >= src_av_frame.nb_samples) {
        // The aligned sample isn't in this frame. Skipping it whole leaves the
        // next one to compute the same aligned sample and land on it.
        // Yes, it may happen that we have to skip more than one frame worth of
        // samples in degenerate cases.
        frame_output.data = torch::stable::empty({out_num_channels, 0});
        return;
      }

      swr_context_.reset(create_swr_context(
          src_sample_format,
          out_sample_format,
          src_sample_rate,
          out_sample_rate,
          src_av_frame,
          out_num_channels));
    }

    converted_av_frame = convert_audio_av_frame_samples(
        swr_context_,
        src_av_frame,
        out_sample_format,
        out_sample_rate,
        out_num_channels,
        static_cast<int>(num_samples_to_skip));
  }
  const AVFrame& av_frame = must_convert ? *converted_av_frame : src_av_frame;

  AVSampleFormat format = static_cast<AVSampleFormat>(av_frame.format);
  STD_TORCH_CHECK(
      format == out_sample_format,
      "Something went wrong, the frame didn't get converted to the desired format. ",
      "Desired format = ",
      av_get_sample_fmt_name(out_sample_format),
      "source format = ",
      av_get_sample_fmt_name(format));

  int num_channels = get_num_channels(av_frame);
  STD_TORCH_CHECK(
      num_channels == out_num_channels,
      "Something went wrong, the frame didn't get converted to the desired ",
      "number of channels = ",
      out_num_channels,
      ". Got ",
      num_channels,
      " instead.");

  auto num_samples = av_frame.nb_samples;

  frame_output.data = torch::stable::empty({num_channels, num_samples});

  if (num_samples > 0) {
    uint8_t* output_channel_data =
        static_cast<uint8_t*>(frame_output.data.mutable_data_ptr());
    auto num_bytes_per_channel = num_samples * av_get_bytes_per_sample(format);
    for (auto channel = 0; channel < num_channels;
         ++channel, output_channel_data += num_bytes_per_channel) {
      std::memcpy(
          output_channel_data,
          av_frame.extended_data[channel],
          num_bytes_per_channel);
    }
  }
}

std::optional<torch::stable::Tensor>
CpuDeviceInterface::maybe_flush_audio_buffers() {
  // When sample rate conversion is involved, swresample buffers some of the
  // samples in-between calls to swr_convert (see the libswresample docs).
  // That's because the last few samples in a given frame require future
  // samples from the next frame to be properly converted. This function
  // flushes out the samples that are stored in swresample's buffers.
  if (!swr_context_) {
    return std::nullopt;
  }
  auto num_remaining_samples = // this is an upper bound
      swr_get_out_samples(swr_context_.get(), 0);

  if (num_remaining_samples == 0) {
    return std::nullopt;
  }

  int num_channels = audio_stream_options_.num_channels.value_or(
      get_num_channels(codec_context_));
  return swr_convert_to_tensor(
      swr_context_,
      /*src_planes=*/nullptr,
      /*num_src_samples=*/0,
      num_channels,
      num_remaining_samples);
}

void CpuDeviceInterface::flush() {
  DeviceInterface::flush();
  swr_context_.reset();
}

std::string CpuDeviceInterface::get_details() {
  return std::string("CPU Device Interface.");
}

UniqueAVFrame CpuDeviceInterface::convert_tensor_to_av_frame_for_encoding(
    const torch::stable::Tensor& frame,
    int frame_index,
    AVCodecContext* codec_context) {
  int in_height = static_cast<int>(frame.sizes()[1]);
  int in_width = static_cast<int>(frame.sizes()[2]);
  AVPixelFormat in_pixel_format = AV_PIX_FMT_GBRP;
  int out_width = codec_context->width;
  int out_height = codec_context->height;
  AVPixelFormat out_pixel_format = codec_context->pix_fmt;

  // Initialize and cache scaling context if it does not exist
  if (!encoding_sws_context_) {
    encoding_sws_context_.reset(sws_getContext(
        in_width,
        in_height,
        in_pixel_format,
        out_width,
        out_height,
        out_pixel_format,
        SWS_BICUBIC, // Used by FFmpeg CLI
        nullptr,
        nullptr,
        nullptr));
    STD_TORCH_CHECK(
        encoding_sws_context_ != nullptr, "Failed to create scaling context");
  }

  UniqueAVFrame av_frame(av_frame_alloc());
  STD_TORCH_CHECK(av_frame != nullptr, "Failed to allocate AVFrame");

  // Set output frame properties
  av_frame->format = out_pixel_format;
  av_frame->width = out_width;
  av_frame->height = out_height;
  av_frame->pts = frame_index;

  int status = av_frame_get_buffer(av_frame.get(), 0);
  STD_TORCH_CHECK(status >= 0, "Failed to allocate frame buffer");

  // Need to convert/scale the frame
  // Create temporary frame with input format
  UniqueAVFrame input_frame(av_frame_alloc());
  STD_TORCH_CHECK(input_frame != nullptr, "Failed to allocate input AVFrame");

  input_frame->format = in_pixel_format;
  input_frame->width = in_width;
  input_frame->height = in_height;

  uint8_t* tensor_data = static_cast<uint8_t*>(frame.mutable_data_ptr());

  int channel_size = in_height * in_width;
  // Since frames tensor is in NCHW, we must use a planar format.
  // FFmpeg only provides AV_PIX_FMT_GBRP for planar RGB,
  // so we reorder RGB -> GBR.
  input_frame->data[0] = tensor_data + channel_size;
  input_frame->data[1] = tensor_data + (2 * channel_size);
  input_frame->data[2] = tensor_data;

  input_frame->linesize[0] = in_width;
  input_frame->linesize[1] = in_width;
  input_frame->linesize[2] = in_width;

  status = sws_scale(
      encoding_sws_context_.get(),
      input_frame->data,
      input_frame->linesize,
      0,
      input_frame->height,
      av_frame->data,
      av_frame->linesize);
  STD_TORCH_CHECK(status == out_height, "sws_scale failed");
  return av_frame;
}

} // namespace facebook::torchcodec
