// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "FFMPEGCommon.h"

#include <cstring>

#include "StableABICompat.h"

extern "C" {
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavutil/pixdesc.h>
}

namespace facebook::torchcodec {

// The AVPixelFormat describing an NVDEC surface for a given source bit depth.
// bit_depth is the source's, and only matters for the 16-bit containers.
// FFmpeg < 6 has no P012LE. P016LE describes the same samples just as
// validly: they're msb-aligned, so a 12-bit surface is a 16-bit one with
// 4 zeroed low bits.
AVPixelFormat nvdec_pix_fmt(NvdecSurface surface, int bit_depth) {
  switch (surface) {
    case NvdecSurface::NV12:
      return AV_PIX_FMT_NV12;
    case NvdecSurface::YUV444:
      return AV_PIX_FMT_YUV444P;
    case NvdecSurface::YUV444_16Bit:
      return AV_PIX_FMT_YUV444P16LE;
    case NvdecSurface::P016:
      if (bit_depth == 10) {
        return AV_PIX_FMT_P010LE;
      }
#if FFMPEG_HAS_P012
      if (bit_depth == 12) {
        return AV_PIX_FMT_P012LE;
      }
#endif
      return AV_PIX_FMT_P016LE;
  }
  return AV_PIX_FMT_NV12;
}

bool is_nvdec_16bit_pix_fmt(int format) {
  return format == AV_PIX_FMT_P010LE || format == AV_PIX_FMT_P016LE ||
#if FFMPEG_HAS_P012
      format == AV_PIX_FMT_P012LE ||
#endif
      false;
}

OutputDtype resolve_output_dtype(
    OutputDtypeConfig output_dtype_config,
    AVPixelFormat pix_fmt) {
  switch (output_dtype_config) {
    case OutputDtypeConfig::UINT8:
      return OutputDtype::UINT8;
    case OutputDtypeConfig::FLOAT32:
      return OutputDtype::FLOAT32;
    case OutputDtypeConfig::AUTO: {
      // TODO_HDR: This is basically our heuristic that defines how we identify
      // HDR videos, we might want to refine it.
      const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(pix_fmt);
      return (desc != nullptr && desc->comp[0].depth > 8) ? OutputDtype::FLOAT32
                                                          : OutputDtype::UINT8;
    }
  }
  return OutputDtype::UINT8;
}

AutoAVPacket::AutoAVPacket() : av_packet_(av_packet_alloc()) {
  STD_TORCH_CHECK(av_packet_ != nullptr, "Couldn't allocate avPacket.");
}

AutoAVPacket::~AutoAVPacket() {
  av_packet_free(&av_packet_);
}

ReferenceAVPacket::ReferenceAVPacket(AutoAVPacket& shared)
    : av_packet_(shared.av_packet_) {}

ReferenceAVPacket::~ReferenceAVPacket() {
  av_packet_unref(av_packet_);
}

AVPacket* ReferenceAVPacket::get() {
  return av_packet_;
}

AVPacket* ReferenceAVPacket::operator->() {
  return av_packet_;
}

AVCodecOnlyUseForCallingAVFindBestStream
make_av_codec_only_use_for_calling_av_find_best_stream(const AVCodec* codec) {
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(59, 18, 100)
  return const_cast<AVCodec*>(codec);
#else
  return codec;
#endif
}

std::string get_ffmpeg_error_string_from_error_code(int error_code) {
  char error_buffer[AV_ERROR_MAX_STRING_SIZE] = {0};
  av_strerror(error_code, error_buffer, AV_ERROR_MAX_STRING_SIZE);
  return std::string(error_buffer);
}

int64_t get_duration(const AVFrame& av_frame) {
#if FFMPEG_HAS_FRAME_DURATION
  return av_frame.duration;
#else
  return av_frame.pkt_duration;
#endif
}

// Some videos aren't properly encoded and do not specify pts values for
// packets, and thus for frames. Unset values correspond to INT64_MIN. When that
// happens, we fall back to the dts value which hopefully exists and is correct.
// Accessing AVFrames' and AVPackets' pts values should **always** go through
// these helpers.
int64_t get_pts_or_dts(ReferenceAVPacket& packet) {
  return packet->pts == INT64_MIN ? packet->dts : packet->pts;
}

int64_t get_pts_or_dts(const AVFrame& av_frame) {
  return av_frame.pts == INT64_MIN ? av_frame.pkt_dts : av_frame.pts;
}

void set_duration(AVFrame& av_frame, int64_t duration) {
#if FFMPEG_HAS_FRAME_DURATION
  av_frame.duration = duration;
#else
  av_frame.pkt_duration = duration;
#endif
}

const int* get_supported_sample_rates(const AVCodec& av_codec) {
  const int* supported_sample_rates = nullptr;
#if FFMPEG_HAS_SUPPORTED_CONFIG
  int num_sample_rates = 0;
  int ret = avcodec_get_supported_config(
      nullptr,
      &av_codec,
      AV_CODEC_CONFIG_SAMPLE_RATE,
      0,
      reinterpret_cast<const void**>(&supported_sample_rates),
      &num_sample_rates);
  if (ret < 0 || supported_sample_rates == nullptr) {
    // Return nullptr to skip validation in validateSampleRate.
    return nullptr;
  }
#else
  supported_sample_rates = av_codec.supported_samplerates;
#endif
  return supported_sample_rates;
}

const AVPixelFormat* get_supported_pixel_formats(const AVCodec& av_codec) {
  const AVPixelFormat* supported_pixel_formats = nullptr;
#if FFMPEG_HAS_SUPPORTED_CONFIG
  int num_pixel_formats = 0;
  int ret = avcodec_get_supported_config(
      nullptr,
      &av_codec,
      AV_CODEC_CONFIG_PIX_FORMAT,
      0,
      reinterpret_cast<const void**>(&supported_pixel_formats),
      &num_pixel_formats);
  if (ret < 0 || supported_pixel_formats == nullptr) {
    STD_TORCH_CHECK(
        false, "Couldn't get supported pixel formats from encoder.");
  }
#else
  supported_pixel_formats = av_codec.pix_fmts;
#endif
  return supported_pixel_formats;
}

const AVSampleFormat* get_supported_output_sample_formats(
    const AVCodec& av_codec) {
  const AVSampleFormat* supported_sample_formats = nullptr;
#if FFMPEG_HAS_SUPPORTED_CONFIG
  int num_sample_formats = 0;
  int ret = avcodec_get_supported_config(
      nullptr,
      &av_codec,
      AV_CODEC_CONFIG_SAMPLE_FORMAT,
      0,
      reinterpret_cast<const void**>(&supported_sample_formats),
      &num_sample_formats);
  if (ret < 0 || supported_sample_formats == nullptr) {
    // Return nullptr to use default output format in
    // findBestOutputSampleFormat.
    return nullptr;
  }
#else
  supported_sample_formats = av_codec.sample_fmts;
#endif
  return supported_sample_formats;
}

#if !FFMPEG_HAS_CH_LAYOUT
// FFmpeg 4 leaves channel_layout unset (0) on some decoded frames even though
// .channels is correct. Everything we feed to swresample needs a real layout,
// so fall back to the default layout for that channel count.
int64_t get_channel_layout(const AVFrame& av_frame) {
  if (av_frame.channel_layout != 0 || av_frame.channels <= 0) {
    return static_cast<int64_t>(av_frame.channel_layout);
  }
  return av_get_default_channel_layout(av_frame.channels);
}
#endif

int get_num_channels(const AVFrame& av_frame) {
#if FFMPEG_HAS_CH_LAYOUT
  return av_frame.ch_layout.nb_channels;
#else
  int num_channels = av_get_channel_layout_nb_channels(av_frame.channel_layout);
  if (num_channels == 0 && av_frame.channels > 0) {
    num_channels = av_frame.channels;
  }
  return num_channels;
#endif
}

int get_num_channels(const SharedAVCodecContext& av_codec_context) {
#if FFMPEG_HAS_CH_LAYOUT
  return av_codec_context->ch_layout.nb_channels;
#else
  return av_codec_context->channels;
#endif
}

int get_num_channels(const AVCodecParameters* codecpar) {
  STD_TORCH_CHECK(codecpar != nullptr, "codecpar is null");
#if FFMPEG_HAS_CH_LAYOUT
  return codecpar->ch_layout.nb_channels;
#else
  return codecpar->channels;
#endif
}

void set_default_channel_layout(
    UniqueAVCodecContext& av_codec_context,
    int num_channels) {
#if FFMPEG_HAS_CH_LAYOUT
  AVChannelLayout channel_layout;
  av_channel_layout_default(&channel_layout, num_channels);
  av_codec_context->ch_layout = channel_layout;
#else
  uint64_t channel_layout = av_get_default_channel_layout(num_channels);
  av_codec_context->channel_layout = channel_layout;
  av_codec_context->channels = num_channels;
#endif
}

void set_default_channel_layout(AVFrame& av_frame, int num_channels) {
#if FFMPEG_HAS_CH_LAYOUT
  AVChannelLayout channel_layout;
  av_channel_layout_default(&channel_layout, num_channels);
  av_frame.ch_layout = channel_layout;
#else
  uint64_t channel_layout = av_get_default_channel_layout(num_channels);
  av_frame.channel_layout = channel_layout;
  av_frame.channels = num_channels;
#endif
}

void validate_num_channels(const AVCodec& av_codec, int num_channels) {
#if FFMPEG_HAS_SUPPORTED_CONFIG
  std::stringstream supported_num_channels;
  const AVChannelLayout* supported_layouts = nullptr;
  int num_layouts = 0;
  int ret = avcodec_get_supported_config(
      nullptr,
      &av_codec,
      AV_CODEC_CONFIG_CHANNEL_LAYOUT,
      0,
      reinterpret_cast<const void**>(&supported_layouts),
      &num_layouts);
  if (ret < 0 || supported_layouts == nullptr) {
    // If we can't validate, we must assume it'll be fine. If not, FFmpeg will
    // eventually raise.
    return;
  }
  for (int i = 0; i < num_layouts; ++i) {
    if (i > 0) {
      supported_num_channels << ", ";
    }
    supported_num_channels << supported_layouts[i].nb_channels;
    if (num_channels == supported_layouts[i].nb_channels) {
      return;
    }
  }
#elif FFMPEG_HAS_CH_LAYOUT
  if (av_codec.ch_layouts == nullptr) {
    // If we can't validate, we must assume it'll be fine. If not, FFmpeg will
    // eventually raise.
    return;
  }
  // FFmpeg doc indicate that the ch_layouts array is terminated by a zeroed
  // layout, so checking for nb_channels == 0 should indicate its end.
  for (auto i = 0; av_codec.ch_layouts[i].nb_channels != 0; ++i) {
    if (num_channels == av_codec.ch_layouts[i].nb_channels) {
      return;
    }
  }
  // At this point it seems that the encoder doesn't support the requested
  // number of channels, so we error out.
  std::stringstream supported_num_channels;
  for (auto i = 0; av_codec.ch_layouts[i].nb_channels != 0; ++i) {
    if (i > 0) {
      supported_num_channels << ", ";
    }
    supported_num_channels << av_codec.ch_layouts[i].nb_channels;
  }
#else // FFmpeg <= 4
  if (av_codec.channel_layouts == nullptr) {
    // can't validate, same as above.
    return;
  }
  for (auto i = 0; av_codec.channel_layouts[i] != 0; ++i) {
    if (num_channels ==
        av_get_channel_layout_nb_channels(av_codec.channel_layouts[i])) {
      return;
    }
  }
  // At this point it seems that the encoder doesn't support the requested
  // number of channels, so we error out.
  std::stringstream supported_num_channels;
  for (auto i = 0; av_codec.channel_layouts[i] != 0; ++i) {
    if (i > 0) {
      supported_num_channels << ", ";
    }
    supported_num_channels << av_get_channel_layout_nb_channels(
        av_codec.channel_layouts[i]);
  }
#endif
  STD_TORCH_CHECK(
      false,
      "Desired number of channels (",
      num_channels,
      ") is not supported by the ",
      "encoder. Supported number of channels are: ",
      supported_num_channels.str(),
      ".");
}

namespace {
#if FFMPEG_HAS_CH_LAYOUT

// Returns:
// - the src_av_frame's channel layout if src_av_frame has out_num_channels
// - the default channel layout with out_num_channels otherwise.
AVChannelLayout get_output_channel_layout(
    int out_num_channels,
    const AVFrame& src_av_frame) {
  AVChannelLayout out_layout;
  if (out_num_channels == get_num_channels(src_av_frame)) {
    out_layout = src_av_frame.ch_layout;
  } else {
    av_channel_layout_default(&out_layout, out_num_channels);
  }
  return out_layout;
}

#else

// Same as above
int64_t get_output_channel_layout(
    int out_num_channels,
    const AVFrame& src_av_frame) {
  int64_t out_layout;
  if (out_num_channels == get_num_channels(src_av_frame)) {
    out_layout = get_channel_layout(src_av_frame);
  } else {
    out_layout = av_get_default_channel_layout(out_num_channels);
  }
  return out_layout;
}
#endif

// Returns av_frame's data planes, advanced by num_samples_to_skip samples:
// one plane per channel for planar formats, a single interleaved one
// otherwise.
std::vector<const uint8_t*> maybe_skip_samples(
    const AVFrame& av_frame,
    int num_samples_to_skip) {
  auto sample_format = static_cast<AVSampleFormat>(av_frame.format);
  bool is_planar = av_sample_fmt_is_planar(sample_format);
  int num_planes = is_planar ? get_num_channels(av_frame) : 1;
  int byte_offset = num_samples_to_skip *
      av_get_bytes_per_sample(sample_format) *
      (is_planar ? 1 : get_num_channels(av_frame));

  std::vector<const uint8_t*> planes(num_planes);
  for (int plane = 0; plane < num_planes; ++plane) {
    // We use extended_data instead of data to support more than 8 channels:
    // data only holds AV_NUM_DATA_POINTERS (8) pointers.
    // https://ffmpeg.org/doxygen/trunk/structAVFrame.html#afca04d808393822625e09b5ba91c6756
    planes[plane] = av_frame.extended_data[plane] + byte_offset;
  }
  return planes;
}
} // namespace

// Sets dst_av_frame' channel layout to get_output_channel_layout(): see doc
// above
void set_channel_layout(
    AVFrame& dst_av_frame,
    const AVFrame& src_av_frame,
    int out_num_channels) {
#if FFMPEG_HAS_CH_LAYOUT
  AVChannelLayout out_layout =
      get_output_channel_layout(out_num_channels, src_av_frame);
  auto status = av_channel_layout_copy(&dst_av_frame.ch_layout, &out_layout);
  STD_TORCH_CHECK(
      status == AVSUCCESS,
      "Couldn't copy channel layout to av_frame: ",
      get_ffmpeg_error_string_from_error_code(status));
#else
  dst_av_frame.channel_layout =
      get_output_channel_layout(out_num_channels, src_av_frame);
  dst_av_frame.channels = out_num_channels;
#endif
}

UniqueAVFrame allocate_av_frame(
    int num_samples,
    int sample_rate,
    int num_channels,
    AVSampleFormat sample_format) {
  auto av_frame = UniqueAVFrame(av_frame_alloc());
  STD_TORCH_CHECK(av_frame != nullptr, "Couldn't allocate AVFrame.");

  av_frame->nb_samples = num_samples;
  av_frame->sample_rate = sample_rate;
  set_default_channel_layout(*av_frame, num_channels);
  av_frame->format = sample_format;
  auto status = av_frame_get_buffer(av_frame.get(), 0);

  STD_TORCH_CHECK(
      status == AVSUCCESS,
      "Couldn't allocate av_frame's buffers: ",
      get_ffmpeg_error_string_from_error_code(status));

  status = av_frame_make_writable(av_frame.get());
  STD_TORCH_CHECK(
      status == AVSUCCESS,
      "Couldn't make AVFrame writable: ",
      get_ffmpeg_error_string_from_error_code(status));
  return av_frame;
}

namespace {
SwrContext* create_swr_context_from_layouts(
    AVSampleFormat src_sample_format,
    AVSampleFormat out_sample_format,
    int src_sample_rate,
    int out_sample_rate,
    const SwrChannelLayout& src_layout,
    const SwrChannelLayout& out_layout) {
  SwrContext* swr_context = nullptr;
  int status = AVSUCCESS;
#if FFMPEG_HAS_CH_LAYOUT
  status = swr_alloc_set_opts2(
      &swr_context,
      // swr_alloc_set_opts2() only became const-correct in FFmpeg 6
      // (libswresample 4.12): before that it asks for non-const layouts that
      // it doesn't modify.
      const_cast<AVChannelLayout*>(&out_layout),
      out_sample_format,
      out_sample_rate,
      const_cast<AVChannelLayout*>(&src_layout),
      src_sample_format,
      src_sample_rate,
      0,
      nullptr);

  STD_TORCH_CHECK(
      status == AVSUCCESS,
      "Couldn't create SwrContext: ",
      get_ffmpeg_error_string_from_error_code(status));
#else
  swr_context = swr_alloc_set_opts(
      nullptr,
      out_layout,
      out_sample_format,
      out_sample_rate,
      src_layout,
      src_sample_format,
      src_sample_rate,
      0,
      nullptr);
#endif

  STD_TORCH_CHECK(swr_context != nullptr, "Couldn't create swr_context");
  status = swr_init(swr_context);
  STD_TORCH_CHECK(
      status == AVSUCCESS,
      "Couldn't initialize SwrContext: ",
      get_ffmpeg_error_string_from_error_code(status),
      ". If the error says 'Invalid argument', it's likely that you are using "
      "a buggy FFmpeg version. FFmpeg4 is known to fail here in some "
      "valid scenarios. Try to upgrade FFmpeg?");
  return swr_context;
}
} // namespace

SwrContext* create_swr_context(
    AVSampleFormat src_sample_format,
    AVSampleFormat out_sample_format,
    int src_sample_rate,
    int out_sample_rate,
    const AVFrame& src_av_frame,
    int out_num_channels) {
  return create_swr_context_from_layouts(
      src_sample_format,
      out_sample_format,
      src_sample_rate,
      out_sample_rate,
#if FFMPEG_HAS_CH_LAYOUT
      src_av_frame.ch_layout,
#else
      get_channel_layout(src_av_frame),
#endif
      get_output_channel_layout(out_num_channels, src_av_frame));
}

SwrContext* create_swr_context(
    AVSampleFormat src_sample_format,
    AVSampleFormat out_sample_format,
    int src_sample_rate,
    int out_sample_rate,
    int src_num_channels,
    int out_num_channels) {
#if FFMPEG_HAS_CH_LAYOUT
  AVChannelLayout src_layout;
  AVChannelLayout out_layout;
  av_channel_layout_default(&src_layout, src_num_channels);
  av_channel_layout_default(&out_layout, out_num_channels);
#else
  int64_t src_layout = av_get_default_channel_layout(src_num_channels);
  int64_t out_layout = av_get_default_channel_layout(out_num_channels);
#endif
  return create_swr_context_from_layouts(
      src_sample_format,
      out_sample_format,
      src_sample_rate,
      out_sample_rate,
      src_layout,
      out_layout);
}

// TODO Other places use the built-in swr_get_out_samples. We should align.
int64_t get_swr_output_num_samples_bound(
    const UniqueSwrContext& swr_context,
    int num_src_samples,
    int src_sample_rate,
    int out_sample_rate) {
  if (src_sample_rate == out_sample_rate) {
    return num_src_samples;
  }
  // Note that this is an upper bound on the number of output samples.
  // `swr_convert()` will likely not produce that many when sample rate
  // conversion is needed: it buffers the last few, because those require
  // future samples. That's why callers must narrow to what it actually
  // returned. We could also use `swr_get_out_samples()`, but empirically
  // `av_rescale_rnd()` gives a tighter bound.
  return av_rescale_rnd(
      swr_get_delay(swr_context.get(), src_sample_rate) + num_src_samples,
      out_sample_rate,
      src_sample_rate,
      AV_ROUND_UP);
}

AVFilterContext* create_av_filter_context_with_options(
    AVFilterGraph* filter_graph,
    const AVFilter* buffer,
    const enum AVPixelFormat output_format) {
  AVFilterContext* av_filter_context = nullptr;
  const char* filter_name = "out";

  enum AVPixelFormat pix_fmts[] = {output_format, AV_PIX_FMT_NONE};

// av_opt_set_int_list was replaced by av_opt_set_array() in FFmpeg 8.
#if LIBAVUTIL_VERSION_MAJOR >= 60 // FFmpeg >= 8
  // Output options like pixel_formats must be set before filter init
  av_filter_context =
      avfilter_graph_alloc_filter(filter_graph, buffer, filter_name);
  STD_TORCH_CHECK(
      av_filter_context != nullptr,
      "Failed to allocate buffer filter context.");

  // When setting pix_fmts, only the first element is used, so nb_elems = 1
  // AV_PIX_FMT_NONE acts as a terminator for the array in av_opt_set_int_list
  int status = av_opt_set_array(
      av_filter_context,
      "pixel_formats",
      AV_OPT_SEARCH_CHILDREN,
      0, // start_elem
      1, // nb_elems
      AV_OPT_TYPE_PIXEL_FMT,
      pix_fmts);
  STD_TORCH_CHECK(
      status >= 0,
      "Failed to set pixel format for buffer filter: ",
      get_ffmpeg_error_string_from_error_code(status));

  status = avfilter_init_str(av_filter_context, nullptr);
  STD_TORCH_CHECK(
      status >= 0,
      "Failed to initialize buffer filter: ",
      get_ffmpeg_error_string_from_error_code(status));
#else // FFmpeg <= 7
  // For older FFmpeg versions, create filter and then set options
  int status = avfilter_graph_create_filter(
      &av_filter_context, buffer, filter_name, nullptr, nullptr, filter_graph);
  STD_TORCH_CHECK(
      status >= 0,
      "Failed to create buffer filter: ",
      get_ffmpeg_error_string_from_error_code(status));

  status = av_opt_set_int_list(
      av_filter_context,
      "pix_fmts",
      pix_fmts,
      AV_PIX_FMT_NONE,
      AV_OPT_SEARCH_CHILDREN);
  STD_TORCH_CHECK(
      status >= 0,
      "Failed to set pixel formats for buffer filter: ",
      get_ffmpeg_error_string_from_error_code(status));
#endif

  return av_filter_context;
}

UniqueAVFrame convert_audio_av_frame_samples(
    const UniqueSwrContext& swr_context,
    const AVFrame& src_av_frame,
    AVSampleFormat out_sample_format,
    int out_sample_rate,
    int out_num_channels,
    int num_samples_to_skip) {
  UniqueAVFrame converted_av_frame(av_frame_alloc());
  STD_TORCH_CHECK(
      converted_av_frame,
      "Could not allocate frame for sample format conversion.");

  converted_av_frame->pts = src_av_frame.pts;
  converted_av_frame->format = static_cast<int>(out_sample_format);

  int num_src_samples = src_av_frame.nb_samples - num_samples_to_skip;
  STD_TORCH_CHECK(
      num_src_samples >= 0,
      "Trying to skip more samples than the frame contains.");
  std::vector<const uint8_t*> src_planes =
      maybe_skip_samples(src_av_frame, num_samples_to_skip);

  converted_av_frame->sample_rate = out_sample_rate;
  // A bound, not the real count, which is why we reset nb_samples after the
  // call to swr_convert() below.
  converted_av_frame
      ->nb_samples = static_cast<int>(get_swr_output_num_samples_bound(
      swr_context, num_src_samples, src_av_frame.sample_rate, out_sample_rate));

  set_channel_layout(*converted_av_frame, src_av_frame, out_num_channels);

  if (converted_av_frame->nb_samples == 0) {
    return converted_av_frame;
  }

  auto status = av_frame_get_buffer(converted_av_frame.get(), 0);
  STD_TORCH_CHECK(
      status == AVSUCCESS,
      "Could not allocate frame buffers for sample format conversion: ",
      get_ffmpeg_error_string_from_error_code(status));

  auto num_converted_samples = swr_convert(
      swr_context.get(),
      converted_av_frame->extended_data,
      converted_av_frame->nb_samples,
      src_planes.data(),
      num_src_samples);
  // numConvertedSamples can be 0 if we're downsampling by a great factor and
  // the first frame doesn't contain a lot of samples. It should be handled
  // properly by the caller.
  STD_TORCH_CHECK(
      num_converted_samples >= 0,
      "Error in swr_convert: ",
      get_ffmpeg_error_string_from_error_code(num_converted_samples));

  // See comment above about nb_samples
  converted_av_frame->nb_samples = num_converted_samples;

  return converted_av_frame;
}

void set_ffmpeg_log_level() {
  auto log_level = AV_LOG_QUIET;
  const char* log_level_env_ptr = std::getenv("TORCHCODEC_FFMPEG_LOG_LEVEL");
  if (log_level_env_ptr != nullptr) {
    std::string log_level_env(log_level_env_ptr);
    if (log_level_env == "QUIET") {
      log_level = AV_LOG_QUIET;
    } else if (log_level_env == "PANIC") {
      log_level = AV_LOG_PANIC;
    } else if (log_level_env == "FATAL") {
      log_level = AV_LOG_FATAL;
    } else if (log_level_env == "ERROR") {
      log_level = AV_LOG_ERROR;
    } else if (log_level_env == "WARNING") {
      log_level = AV_LOG_WARNING;
    } else if (log_level_env == "INFO") {
      log_level = AV_LOG_INFO;
    } else if (log_level_env == "VERBOSE") {
      log_level = AV_LOG_VERBOSE;
    } else if (log_level_env == "DEBUG") {
      log_level = AV_LOG_DEBUG;
    } else if (log_level_env == "TRACE") {
      log_level = AV_LOG_TRACE;
    } else {
      STD_TORCH_CHECK(
          false,
          "Invalid TORCHCODEC_FFMPEG_LOG_LEVEL: ",
          log_level_env,
          ". Use e.g. 'QUIET', 'PANIC', 'VERBOSE', etc.");
    }
  }
  av_log_set_level(log_level);
}

AVIOContext* avio_alloc_context(
    uint8_t* buffer,
    int buffer_size,
    int write_flag,
    void* opaque,
    AVIOReadFunction read_packet,
    AVIOWriteFunction write_packet,
    AVIOSeekFunction seek) {
  // Qualified with :: to call FFmpeg's avio_alloc_context, not this same-named
  // wrapper in the facebook::torchcodec namespace (which would recurse).
  return ::avio_alloc_context(
      buffer,
      buffer_size,
      write_flag,
      opaque,
      read_packet,
// The buf parameter of the write function is not const before FFmpeg 7.
#if LIBAVFILTER_VERSION_MAJOR >= 10 // FFmpeg >= 7
      write_packet,
#else
      reinterpret_cast<AVIOWriteFunctionOld>(write_packet),
#endif
      seek);
}

double pts_to_seconds(int64_t pts, const AVRational& time_base) {
  // To perform the multiplication before the division, av_q2d is not used
  return static_cast<double>(pts) * time_base.num / time_base.den;
}

int64_t seconds_to_closest_pts(double seconds, const AVRational& time_base) {
  return static_cast<int64_t>(
      std::round(seconds * time_base.den / time_base.num));
}

int64_t compute_safe_duration(
    const AVRational& frame_rate,
    const AVRational& time_base) {
  if (frame_rate.num <= 0 || frame_rate.den <= 0 || time_base.num <= 0 ||
      time_base.den <= 0) {
    return 0;
  } else {
    return (static_cast<int64_t>(frame_rate.den) * time_base.den) /
        (static_cast<int64_t>(time_base.num) * frame_rate.num);
  }
}

const int32_t* get_display_matrix_from_stream(const AVStream* av_stream) {
  // av_stream_get_side_data() was deprecated in FFmpeg 6.0, but its replacement
  // (av_packet_side_data_get() + codecpar->coded_side_data) is only available
  // from FFmpeg 6.1. We need some #pragma magic to silence the deprecation
  // warning which our compile chain would otherwise treat as an error.
  if (av_stream == nullptr) {
    return nullptr;
  }

  const int32_t* display_matrix = nullptr;

// FFmpeg >= 6.1: Use codecpar->coded_side_data
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(60, 31, 100)
  const AVPacketSideData* side_data = av_packet_side_data_get(
      av_stream->codecpar->coded_side_data,
      av_stream->codecpar->nb_coded_side_data,
      AV_PKT_DATA_DISPLAYMATRIX);
  if (side_data != nullptr) {
    display_matrix = reinterpret_cast<const int32_t*>(side_data->data);
  }
#elif LIBAVFORMAT_VERSION_MAJOR >= 60 // FFmpeg 6.0
  // FFmpeg 6.0: Use av_stream_get_side_data (deprecated but still available)
  // Suppress deprecation warning for this specific call
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  size_t side_data_size = 0;
  const uint8_t* side_data = av_stream_get_side_data(
      av_stream, AV_PKT_DATA_DISPLAYMATRIX, &side_data_size);
#pragma GCC diagnostic pop
  if (side_data != nullptr) {
    display_matrix = reinterpret_cast<const int32_t*>(side_data);
  }
#else
  // FFmpeg < 6: Use av_stream_get_side_data.
  // The size parameter type changed from int* (FFmpeg 4) to size_t* (FFmpeg 5)
#if LIBAVFORMAT_VERSION_MAJOR >= 59 // FFmpeg 5
  size_t side_data_size = 0;
#else // FFmpeg 4
  int side_data_size = 0;
#endif
  const uint8_t* side_data = av_stream_get_side_data(
      av_stream, AV_PKT_DATA_DISPLAYMATRIX, &side_data_size);
  if (side_data != nullptr) {
    display_matrix = reinterpret_cast<const int32_t*>(side_data);
  }
#endif

  return display_matrix;
}

namespace {
std::optional<double> get_rotation_from_display_matrix(
    const int32_t* display_matrix) {
  if (display_matrix == nullptr) {
    return std::nullopt;
  }

  // av_display_rotation_get returns the rotation angle in degrees needed to
  // rotate the video counter-clockwise to make it upright.
  // Returns NaN if the matrix is invalid.
  double rotation = av_display_rotation_get(display_matrix);

  // Check for invalid matrix
  if (std::isnan(rotation)) {
    return std::nullopt;
  }

  return rotation;
}
} // namespace

std::optional<double> get_rotation_from_stream(const AVStream* av_stream) {
  return get_rotation_from_display_matrix(
      get_display_matrix_from_stream(av_stream));
}

std::optional<double> get_rotation_from_frame(const AVFrame& av_frame) {
  const AVFrameSideData* side_data =
      av_frame_get_side_data(&av_frame, AV_FRAME_DATA_DISPLAYMATRIX);
  if (side_data == nullptr) {
    return std::nullopt;
  }
  return get_rotation_from_display_matrix(
      reinterpret_cast<const int32_t*>(side_data->data));
}

void set_display_matrix_on_frame(
    AVFrame& av_frame,
    const int32_t* display_matrix) {
  // The frame may already carry a display matrix of its own: FFmpeg propagates
  // the container's from 6.1 on, and the H.264/HEVC decoders derive one from a
  // display-orientation SEI. Since av_frame_new_side_data() appends rather than
  // replaces, and av_frame_get_side_data() returns the first match, not
  // clearing first would leave whatever the decoder attached in charge and make
  // ours dead weight - so which matrix wins would depend on the FFmpeg version
  // and the codec. Clearing keeps it always the container's, which is the one
  // the SingleStreamDecoder applies.
  //
  // Note that no test covers this: for our assets the decoder's matrix, when
  // there is one, is the container's, so dropping this line changes nothing
  // observable. It would take a stream whose SEI disagrees with its container.
  av_frame_remove_side_data(&av_frame, AV_FRAME_DATA_DISPLAYMATRIX);
  if (display_matrix == nullptr) {
    return;
  }

  constexpr size_t kDisplayMatrixSize = 9 * sizeof(int32_t);
  AVFrameSideData* side_data = av_frame_new_side_data(
      &av_frame, AV_FRAME_DATA_DISPLAYMATRIX, kDisplayMatrixSize);
  STD_TORCH_CHECK(
      side_data != nullptr, "Failed to allocate display matrix side data");
  std::memcpy(side_data->data, display_matrix, kDisplayMatrixSize);
}

SwsConfig::SwsConfig(
    int input_width,
    int input_height,
    AVPixelFormat input_format,
    AVColorSpace input_colorspace,
    int output_width,
    int output_height,
    AVPixelFormat output_format,
    AVColorRange output_color_range)
    : input_width(input_width),
      input_height(input_height),
      input_format(input_format),
      input_colorspace(input_colorspace),
      output_width(output_width),
      output_height(output_height),
      output_format(output_format),
      output_color_range(output_color_range) {}

bool SwsConfig::operator==(const SwsConfig& other) const {
  return input_width == other.input_width &&
      input_height == other.input_height &&
      input_format == other.input_format &&
      input_colorspace == other.input_colorspace &&
      output_width == other.output_width &&
      output_height == other.output_height &&
      output_format == other.output_format &&
      output_color_range == other.output_color_range;
}

bool SwsConfig::operator!=(const SwsConfig& other) const {
  return !(*this == other);
}

UniqueSwsContext create_sws_context(
    const SwsConfig& sws_config,
    int sws_flags) {
  SwsContext* sws_context = sws_getContext(
      sws_config.input_width,
      sws_config.input_height,
      sws_config.input_format,
      sws_config.output_width,
      sws_config.output_height,
      sws_config.output_format,
      sws_flags,
      nullptr,
      nullptr,
      nullptr);
  STD_TORCH_CHECK(sws_context, "sws_getContext() returned nullptr");

  int* inv_table = nullptr;
  int* table = nullptr;
  int src_range, dst_range, brightness, contrast, saturation;
  int ret = sws_getColorspaceDetails(
      sws_context,
      &inv_table,
      &src_range,
      &table,
      &dst_range,
      &brightness,
      &contrast,
      &saturation);
  STD_TORCH_CHECK(ret != -1, "sws_getColorspaceDetails returned -1");

  // swscale spells a range as an int: 1 is full, 0 is limited. FFmpeg names
  // those AVCOL_RANGE_JPEG and AVCOL_RANGE_MPEG, after the two worlds they come
  // from - JPEG is the full one.
  if (sws_config.output_color_range != AVCOL_RANGE_UNSPECIFIED) {
    dst_range = sws_config.output_color_range == AVCOL_RANGE_JPEG;
  }

  const int* colorspace_table =
      sws_getCoefficients(sws_config.input_colorspace);
  ret = sws_setColorspaceDetails(
      sws_context,
      colorspace_table,
      src_range,
      colorspace_table,
      dst_range,
      brightness,
      contrast,
      saturation);
  STD_TORCH_CHECK(ret != -1, "sws_setColorspaceDetails returned -1");

  return UniqueSwsContext(sws_context);
}

} // namespace facebook::torchcodec
