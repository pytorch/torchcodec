// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/torchcodec/decoders/_core/FFMPEGCommon.h"

#include <c10/util/Exception.h>

namespace facebook::torchcodec {

AutoAVPacket::AutoAVPacket() : avPacket_(av_packet_alloc()) {
  TORCH_CHECK(avPacket_ != nullptr, "Couldn't allocate avPacket.");
}

AutoAVPacket::~AutoAVPacket() {
  av_packet_free(&avPacket_);
}

ReferenceAVPacket::ReferenceAVPacket(AutoAVPacket& shared)
    : avPacket_(shared.avPacket_) {}

ReferenceAVPacket::~ReferenceAVPacket() {
  av_packet_unref(avPacket_);
}

AVPacket* ReferenceAVPacket::get() {
  return avPacket_;
}

AVPacket* ReferenceAVPacket::operator->() {
  return avPacket_;
}

AVCodecOnlyUseForCallingAVFindBestStream
makeAVCodecOnlyUseForCallingAVFindBestStream(const AVCodec* codec) {
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(59, 18, 100)
  return const_cast<AVCodec*>(codec);
#else
  return codec;
#endif
}

std::string getFFMPEGErrorStringFromErrorCode(int errorCode) {
  char errorBuffer[AV_ERROR_MAX_STRING_SIZE] = {0};
  av_strerror(errorCode, errorBuffer, AV_ERROR_MAX_STRING_SIZE);
  return std::string(errorBuffer);
}

int64_t getDuration(const UniqueAVFrame& frame) {
  return getDuration(frame.get());
}

int64_t getDuration(const AVFrame* frame) {
#if LIBAVUTIL_VERSION_MAJOR < 58
  return frame->pkt_duration;
#else
  return frame->duration;
#endif
}

int getNumChannels(const UniqueAVFrame& avFrame) {
#if LIBAVFILTER_VERSION_MAJOR > 8 || \
    (LIBAVFILTER_VERSION_MAJOR == 8 && LIBAVFILTER_VERSION_MINOR >= 44)
  return avFrame->ch_layout.nb_channels;
#else
  return av_get_channel_layout_nb_channels(avFrame->channel_layout);
#endif
}

int getNumChannels(const UniqueAVCodecContext& avCodecContext) {
#if LIBAVFILTER_VERSION_MAJOR > 8 || \
    (LIBAVFILTER_VERSION_MAJOR == 8 && LIBAVFILTER_VERSION_MINOR >= 44)
  return avCodecContext->ch_layout.nb_channels;
#else
  return avCodecContext->channels;
#endif
}

void setChannelLayout(
    UniqueAVFrame& dstAVFrame,
    const UniqueAVFrame& srcAVFrame) {
#if LIBAVFILTER_VERSION_MAJOR > 7 // FFmpeg > 4
  dstAVFrame->ch_layout = srcAVFrame->ch_layout;
#else
  dstAVFrame->channel_layout = srcAVFrame->channel_layout;
#endif
}

SwrContext* allocateSwrContext(
    UniqueAVCodecContext& avCodecContext,
    int sampleRate,
    AVSampleFormat sourceSampleFormat,
    AVSampleFormat desiredSampleFormat) {
  SwrContext* swrContext = nullptr;
#if LIBAVFILTER_VERSION_MAJOR > 7 // FFmpeg > 4
  AVChannelLayout layout = avCodecContext->ch_layout;
  auto status = swr_alloc_set_opts2(
      &swrContext,
      &layout,
      desiredSampleFormat,
      sampleRate,
      &layout,
      sourceSampleFormat,
      sampleRate,
      0,
      nullptr);

  TORCH_CHECK(
      status == AVSUCCESS,
      "Couldn't create SwrContext: ",
      getFFMPEGErrorStringFromErrorCode(status));
#else
  int64_t layout = static_cast<int64_t>(avCodecContext->channel_layout);
  swrContext = swr_alloc_set_opts(
      nullptr,
      layout,
      desiredSampleFormat,
      sampleRate,
      layout,
      sourceSampleFormat,
      sampleRate,
      0,
      nullptr);
#endif

  TORCH_CHECK(swrContext != nullptr, "Couldn't create swrContext");
  return swrContext;
}

} // namespace facebook::torchcodec
