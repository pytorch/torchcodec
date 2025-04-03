// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/torchcodec/_core/FFMPEGCommon.h"

#include <c10/util/Exception.h>

namespace facebook::torchcodec {

AutoAVPacket::AutoAVPacket() : avpacket_(avpacket_alloc()) {
  TORCH_CHECK(avPacket_ != nullptr, "Couldn't allocate avpacket.");
}

AutoAVPacket::~AutoAVPacket() {
  avpacket_free(&avpacket_);
}

ReferenceAVPacket::ReferenceAVPacket(AutoAVPacket& shared)
    : avpacket_(shared.avpacket_) {}

ReferenceAVPacket::~ReferenceAVPacket() {
  avpacket_unref(avpacket_);
}

AVPacket* ReferenceAVPacket::get() {
  return avpacket_;
}

AVPacket* ReferenceAVPacket::operator->() {
  return avpacket_;
}

AVCodecOnlyUseForCallingAVFindBestStream
make_avcodec_only_use_for_calling_avfind_best_stream(const AVCodec* codec) {
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(59, 18, 100)
  return const_cast<_avcodec*>(codec);
#else
  return codec;
#endif
}

std::string get_ffmpeg_error_string_from_error_code(int error_code) {
  char error_buffer[AV_ERROR_MAX_STRING_SIZE] = {0};
  av_strerror(error_code, error_buffer, AV_ERROR_MAX_STRING_SIZE);
  return std::string(error_buffer);
}

int64_t get_duration(const UniqueAVFrame& avframe) {
#if LIBAVUTIL_VERSION_MAJOR < 58
  return avframe->pkt_duration;
#else
  return avframe->duration;
#endif
}

int get_num_channels(const UniqueAVFrame& avframe) {
#if LIBAVFILTER_VERSION_MAJOR > 8 || \
    (LIBAVFILTER_VERSION_MAJOR == 8 && LIBAVFILTER_VERSION_MINOR >= 44)
  return avframe->ch_layout.nb_channels;
#else
  return av_get_channel_layout_nb_channels(avframe->channel_layout);
#endif
}

int get_num_channels(const UniqueAVCodecContext& av_codec_context) {
#if LIBAVFILTER_VERSION_MAJOR > 8 || \
    (LIBAVFILTER_VERSION_MAJOR == 8 && LIBAVFILTER_VERSION_MINOR >= 44)
  return av_codec_context->ch_layout.nb_channels;
#else
  return av_codec_context->channels;
#endif
}

void set_channel_layout(
    UniqueAVFrame& dst_avframe,
    const UniqueAVFrame& src_avframe) {
#if LIBAVFILTER_VERSION_MAJOR > 7 // FFmpeg > 4
  dst_avframe->ch_layout = src_avframe->ch_layout;
#else
  dst_avframe->channel_layout = src_avframe->channel_layout;
#endif
}

SwrContext* allocate_swr_context(
    UniqueAVCodecContext& av_codec_context,
    AVSampleFormat source_sample_format,
    AVSampleFormat desired_sample_format,
    int source_sample_rate,
    int desired_sample_rate) {
  SwrContext* swr_context = nullptr;
#if LIBAVFILTER_VERSION_MAJOR > 7 // FFmpeg > 4
  AVChannelLayout layout = av_codec_context->ch_layout;
  auto status = swr_alloc_set_opts2(
      &swrContext,
      &layout,
      desired_sample_format,
      desired_sample_rate,
      &layout,
      source_sample_format,
      source_sample_rate,
      0,
      nullptr);

  TORCH_CHECK(
      status == AVSUCCESS,
      "Couldn't create SwrContext: ",
      get_ffmpeg_error_string_from_error_code(status));
#else
  int64_t layout = static_cast<int64_t>(av_codec_context->channel_layout);
  swr_context = swr_alloc_set_opts(
      nullptr,
      layout,
      desired_sample_format,
      desired_sample_rate,
      layout,
      source_sample_format,
      source_sample_rate,
      0,
      nullptr);
#endif

  TORCH_CHECK(swrContext != nullptr, "Couldn't create swr_context");
  return swr_context;
}

} // namespace facebook::torchcodec
