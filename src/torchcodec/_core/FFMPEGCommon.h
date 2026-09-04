// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "StreamOptions.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavcodec/bsf.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersrc.h>
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
#include <libavutil/audio_fifo.h>
#include <libavutil/avutil.h>
#include <libavutil/dict.h>
#include <libavutil/display.h>
#include <libavutil/file.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libavutil/pixfmt.h>
#include <libavutil/version.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
}

// FFmpeg 5.1 replaced the .channels + .channel_layout pair on AVFrame and
// AVCodecContext with a single AVChannelLayout .ch_layout, and added the
// av_channel_layout_* / swr_alloc_set_opts2() APIs that go with it.
// libavutil 57.24 is the real marker, but libavfilter 8.44 is the equivalent
// and is what this codebase has always tested against.
#if LIBAVFILTER_VERSION_MAJOR > 8 || \
    (LIBAVFILTER_VERSION_MAJOR == 8 && LIBAVFILTER_VERSION_MINOR >= 44)
#define FFMPEG_HAS_CH_LAYOUT 1
#else
#define FFMPEG_HAS_CH_LAYOUT 0
#endif

// FFmpeg 6 added AV_PIX_FMT_P012, the 12-bit sibling of P010.
#if LIBAVUTIL_VERSION_MAJOR >= 58
#define FFMPEG_HAS_P012 1
#else
#define FFMPEG_HAS_P012 0
#endif

// FFmpeg 7.1 added avcodec_get_supported_config(), replacing the codec's
// pix_fmts / sample_fmts / supported_samplerates / ch_layouts arrays.
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(61, 13, 100)
#define FFMPEG_HAS_SUPPORTED_CONFIG 1
#else
#define FFMPEG_HAS_SUPPORTED_CONFIG 0
#endif

// FFmpeg 6 renamed AVFrame.pkt_duration to AVFrame.duration.
#if LIBAVUTIL_VERSION_MAJOR < 58
#define FFMPEG_HAS_FRAME_DURATION 0
#else
#define FFMPEG_HAS_FRAME_DURATION 1
#endif

namespace facebook::torchcodec {

// Mirrors cudaVideoSurfaceFormat. Needed here because we don't want to include
// the CUDA headers in this file, and it is necessary for
// ffmpeg-version-dependent helpers like nvdec_pix_fmt() and
// is_nvdec_16bit_pix_fmt().
enum class NvdecSurface {
  NV12,
  P016,
  YUV444,
  YUV444_16Bit,
};

AVPixelFormat nvdec_pix_fmt(NvdecSurface surface, int bit_depth);

bool is_nvdec_16bit_pix_fmt(int format);

OutputDtype resolve_output_dtype(
    OutputDtypeConfig output_dtype_config,
    AVPixelFormat pix_fmt);

// FFMPEG uses special delete functions for some structures. These template
// functions are used to pass into unique_ptr as custom deleters so we can
// wrap FFMPEG structs with unique_ptrs for ease of use.
template <typename T, typename R, R (*Fn)(T**)>
struct Deleterp {
  inline void operator()(T* p) const {
    if (p) {
      Fn(&p);
    }
  }
};

template <typename T, typename R, R (*Fn)(void*)>
struct Deleterv {
  inline void operator()(T* p) const {
    if (p) {
      Fn(&p);
    }
  }
};

template <typename T, typename R, R (*Fn)(T*)>
struct Deleter {
  inline void operator()(T* p) const {
    if (p) {
      Fn(p);
    }
  }
};

// Unique pointers for FFMPEG structures.
using UniqueDecodingAVFormatContext = std::unique_ptr<
    AVFormatContext,
    Deleterp<AVFormatContext, void, avformat_close_input>>;
using UniqueEncodingAVFormatContext = std::unique_ptr<
    AVFormatContext,
    Deleter<AVFormatContext, void, avformat_free_context>>;
using UniqueAVCodecContext = std::unique_ptr<
    AVCodecContext,
    Deleterp<AVCodecContext, void, avcodec_free_context>>;
using SharedAVCodecContext = std::shared_ptr<AVCodecContext>;

// create SharedAVCodecContext with custom deleter
inline SharedAVCodecContext make_shared_av_codec_context(AVCodecContext* ctx) {
  return SharedAVCodecContext(
      ctx, Deleterp<AVCodecContext, void, avcodec_free_context>{});
}

using UniqueAVFrame =
    std::unique_ptr<AVFrame, Deleterp<AVFrame, void, av_frame_free>>;
using UniqueAVPacket =
    std::unique_ptr<AVPacket, Deleterp<AVPacket, void, av_packet_free>>;
using UniqueAVFilterGraph = std::unique_ptr<
    AVFilterGraph,
    Deleterp<AVFilterGraph, void, avfilter_graph_free>>;
using UniqueAVFilterInOut = std::unique_ptr<
    AVFilterInOut,
    Deleterp<AVFilterInOut, void, avfilter_inout_free>>;
using UniqueAVIOContext = std::
    unique_ptr<AVIOContext, Deleterp<AVIOContext, void, avio_context_free>>;
using UniqueSwsContext =
    std::unique_ptr<SwsContext, Deleter<SwsContext, void, sws_freeContext>>;
using UniqueSwrContext =
    std::unique_ptr<SwrContext, Deleterp<SwrContext, void, swr_free>>;
using UniqueAVAudioFifo = std::
    unique_ptr<AVAudioFifo, Deleter<AVAudioFifo, void, av_audio_fifo_free>>;
using UniqueAVBSFContext =
    std::unique_ptr<AVBSFContext, Deleterp<AVBSFContext, void, av_bsf_free>>;
using UniqueAVBufferRef =
    std::unique_ptr<AVBufferRef, Deleterp<AVBufferRef, void, av_buffer_unref>>;
using UniqueAVBufferSrcParameters = std::unique_ptr<
    AVBufferSrcParameters,
    Deleterv<AVBufferSrcParameters, void, av_freep>>;

// Wrapper class for AVDictionary, similar to unique_ptr, to support FFmpeg's
// functions that require a double-pointer to AVDictionary, such as av_dict_set.
// https://ffmpeg.org/doxygen/trunk/group__lavu__dict.html#ga8d9c2de72b310cef8e6a28c9cd3acbbe
class UniqueAVDictionary {
 private:
  AVDictionary* dict_ = nullptr;

 public:
  UniqueAVDictionary() = default;

  ~UniqueAVDictionary() {
    if (dict_) {
      av_dict_free(&dict_);
    }
  }

  // Explicitly delete copy operator similar to unique_ptr
  UniqueAVDictionary(const UniqueAVDictionary&) = delete;
  UniqueAVDictionary& operator=(const UniqueAVDictionary&) = delete;
  // Explicitly delete move operator, as it is not needed at this time.
  UniqueAVDictionary(UniqueAVDictionary&&) = delete;
  UniqueAVDictionary& operator=(UniqueAVDictionary&&) = delete;

  // FFmpeg's AVDictionary functions require a AVDictionary** argument.
  // However, unique_ptr's get() function returns a **temporary** pointer to the
  // object, so we cannot get a pointer to the internal AVDictionary pointer.
  // As a result, we implement getAddress() to return a pointer to the internal
  // AVDictionary pointer.
  AVDictionary** get_address() {
    return &dict_;
  }
};

// These 2 classes share the same underlying AVPacket object. They are meant to
// be used in tandem, like so:
//
// AutoAVPacket autoAVPacket; // <-- malloc for AVPacket happens here
// while(...){
//   ReferenceAVPacket packet(autoAVPacket);
//   av_read_frame(..., packet.get());  <-- av_packet_ref() called by FFmpeg
// } <-- av_packet_unref() called here
//
// This achieves a few desirable things:
// - Memory allocation of the underlying AVPacket happens only once, when
//   autoAVPacket is created.
// - av_packet_free() is called when autoAVPacket gets out of scope
// - av_packet_unref() is automatically called when needed, i.e. at the end of
//   each loop iteration (or when hitting break / continue). This prevents the
//   risk of us forgetting to call it.
class AutoAVPacket {
  friend class ReferenceAVPacket;

 private:
  AVPacket* av_packet_;

 public:
  AutoAVPacket();
  AutoAVPacket(const AutoAVPacket& other) = delete;
  AutoAVPacket& operator=(const AutoAVPacket& other) = delete;
  ~AutoAVPacket();
};

class ReferenceAVPacket {
 private:
  AVPacket* av_packet_;

 public:
  explicit ReferenceAVPacket(AutoAVPacket& shared);
  ReferenceAVPacket(const ReferenceAVPacket& other) = delete;
  ReferenceAVPacket& operator=(const ReferenceAVPacket& other) = delete;
  ~ReferenceAVPacket();
  AVPacket* get();
  AVPacket& operator*();
  AVPacket* operator->();
};

// av_find_best_stream is not const-correct before commit:
// https://github.com/FFmpeg/FFmpeg/commit/46dac8cf3d250184ab4247809bc03f60e14f4c0c
// which was released in FFMPEG version=5.0.3
// with libavcodec's version=59.18.100
// (https://www.ffmpeg.org/olddownload.html).
// Note that the alias is so-named so that it is only used when interacting with
// av_find_best_stream(). It is not needed elsewhere.
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(59, 18, 100)
using AVCodecOnlyUseForCallingAVFindBestStream = AVCodec*;
#else
using AVCodecOnlyUseForCallingAVFindBestStream = const AVCodec*;
#endif

AVCodecOnlyUseForCallingAVFindBestStream
make_av_codec_only_use_for_calling_av_find_best_stream(const AVCodec* codec);

// Success code from FFMPEG is just a 0. We define it to make the code more
// readable.
const int AVSUCCESS = 0;

// Returns the FFMPEG error as a string using the provided `errorCode`.
std::string get_ffmpeg_error_string_from_error_code(int error_code);

// Returns duration from the frame. Abstracted into a function because the
// struct member representing duration has changed across the versions we
// support.
int64_t get_duration(const AVFrame& frame);
void set_duration(AVFrame& frame, int64_t duration);

// pts accessors that fall back to dts when pts is unset (INT64_MIN). See the
// definitions for details.
int64_t get_pts_or_dts(ReferenceAVPacket& packet);
int64_t get_pts_or_dts(const AVFrame& av_frame);

const int* get_supported_sample_rates(const AVCodec& av_codec);
const AVSampleFormat* get_supported_output_sample_formats(
    const AVCodec& av_codec);
const AVPixelFormat* get_supported_pixel_formats(const AVCodec& av_codec);

int get_num_channels(const AVFrame& av_frame);
int get_num_channels(const SharedAVCodecContext& av_codec_context);
int get_num_channels(const AVCodecParameters* codecpar);

void set_default_channel_layout(
    UniqueAVCodecContext& av_codec_context,
    int num_channels);

void set_default_channel_layout(AVFrame& av_frame, int num_channels);

void validate_num_channels(const AVCodec& av_codec, int num_channels);

void set_channel_layout(
    AVFrame& dst_av_frame,
    const AVFrame& src_av_frame,
    int desired_num_channels);

UniqueAVFrame allocate_av_frame(
    int num_samples,
    int sample_rate,
    int num_channels,
    AVSampleFormat sample_format);

// How a channel layout is spelled, which changed in FFmpeg 5.
#if FFMPEG_HAS_CH_LAYOUT
using SwrChannelLayout = AVChannelLayout;
#else
using SwrChannelLayout = int64_t;
#endif

SwrContext* create_swr_context(
    AVSampleFormat src_sample_format,
    AVSampleFormat desired_sample_format,
    int src_sample_rate,
    int desired_sample_rate,
    const AVFrame& src_av_frame,
    int desired_num_channels);

// Same, for callers that have samples rather than an AVFrame, and so only know
// how many channels there are. Both layouts are the default one for their
// channel count, which is what the AVFrame overload also falls back to when the
// count changes.
SwrContext* create_swr_context(
    AVSampleFormat src_sample_format,
    AVSampleFormat desired_sample_format,
    int src_sample_rate,
    int desired_sample_rate,
    int src_num_channels,
    int desired_num_channels);

// Upper bound on the number of samples swr_convert() can emit for
// num_src_samples of input, including whatever it still holds buffered. It is
// only a bound: narrow the output down to what swr_convert() returned.
int64_t get_swr_output_num_samples_bound(
    const UniqueSwrContext& swr_context,
    int num_src_samples,
    int src_sample_rate,
    int out_sample_rate);

// Converts, if needed:
// - sample format
// - sample rate
// - number of channels.
// createSwrContext must have been previously called with matching parameters.
UniqueAVFrame convert_audio_av_frame_samples(
    const UniqueSwrContext& swr_context,
    const AVFrame& src_av_frame,
    AVSampleFormat desired_sample_format,
    int desired_sample_rate,
    int desired_num_channels,
    int num_samples_to_skip = 0);

// Returns true if sws_scale can handle unaligned data.
bool can_sws_scale_handle_unaligned_data();

void set_ffmpeg_log_level();

// These signatures are defined by FFmpeg.
using AVIOReadFunction = int (*)(void*, uint8_t*, int);
using AVIOWriteFunction = int (*)(void*, const uint8_t*, int); // FFmpeg >= 7
using AVIOWriteFunctionOld = int (*)(void*, uint8_t*, int); // FFmpeg < 7
using AVIOSeekFunction = int64_t (*)(void*, int64_t, int);

AVIOContext* avio_alloc_context(
    uint8_t* buffer,
    int buffer_size,
    int write_flag,
    void* opaque,
    AVIOReadFunction read_packet,
    AVIOWriteFunction write_packet,
    AVIOSeekFunction seek);

double pts_to_seconds(int64_t pts, const AVRational& time_base);
int64_t seconds_to_closest_pts(double seconds, const AVRational& time_base);
int64_t compute_safe_duration(
    const AVRational& frame_rate,
    const AVRational& time_base);

// Extracts the rotation angle in degrees from the stream's display matrix
// side data. The display matrix is used to specify how the video should be
// rotated for correct display.
std::optional<double> get_rotation_from_stream(const AVStream* av_stream);

// Same, from a frame's own display matrix side data.
std::optional<double> get_rotation_from_frame(const AVFrame& av_frame);

// The stream's raw display matrix, or nullptr when it has none. Points into the
// stream's side data, so it lives as long as the stream does.
const int32_t* get_display_matrix_from_stream(const AVStream* av_stream);

// Makes `av_frame` carry `display_matrix` (or no display matrix at all, for
// nullptr), replacing whatever it had.
void set_display_matrix_on_frame(
    AVFrame& av_frame,
    const int32_t* display_matrix);

AVFilterContext* create_av_filter_context_with_options(
    AVFilterGraph* filter_graph,
    const AVFilter* buffer,
    const enum AVPixelFormat output_format);

struct SwsConfig {
  int input_width = 0;
  int input_height = 0;
  AVPixelFormat input_format = AV_PIX_FMT_NONE;
  AVColorSpace input_colorspace = AVCOL_SPC_UNSPECIFIED;
  int output_width = 0;
  int output_height = 0;
  AVPixelFormat output_format = AV_PIX_FMT_NONE;
  AVColorRange output_color_range = AVCOL_RANGE_UNSPECIFIED;

  SwsConfig() = default;
  SwsConfig(
      int input_width,
      int input_height,
      AVPixelFormat input_format,
      AVColorSpace input_colorspace,
      int output_width,
      int output_height,
      AVPixelFormat output_format,
      AVColorRange output_color_range = AVCOL_RANGE_UNSPECIFIED);

  bool operator==(const SwsConfig& other) const;
  bool operator!=(const SwsConfig& other) const;
};

// Utility functions for swscale context management
UniqueSwsContext create_sws_context(const SwsConfig& sws_config, int sws_flags);

} // namespace facebook::torchcodec
